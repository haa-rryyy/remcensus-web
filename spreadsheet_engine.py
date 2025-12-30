import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re
from difflib import SequenceMatcher
import logging
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data (do this once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    import Levenshtein

    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# 1. DATA MODELS
# ============================================================================


class MatchStrategy(Enum):
    """Strategies for matching entries"""

    EXACT = "exact"
    FUZZY = "fuzzy"
    LEVENSHTEIN = "levenshtein"
    KEYWORD = "keyword"
    PATTERN = "pattern"


@dataclass
class MatchResult:
    """Result of a single match"""

    value: str  # The matched value
    column: str  # Which column it came from
    score: float  # Match confidence (0-1)
    strategy: MatchStrategy  # Which strategy found it
    row_index: int  # Row in the dataframe

    def __lt__(self, other):
        """Sort by score descending"""
        return self.score > other.score


@dataclass
class RelationalLookupResult:
    """Result of a relational lookup"""

    source_column: str
    source_value: str
    target_column: str
    target_value: str
    row_index: int


# ============================================================================
# 2. TEXT PROCESSING UTILITIES
# ============================================================================


class TextProcessor:
    """Handles text normalization and cleaning"""

    # Initialize stemmer
    stemmer = PorterStemmer()

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for comparison"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase, strip whitespace
        text = text.lower().strip()
        # Handle common OCR artifacts
        text = text.replace(""", "'").replace(""", "'")  # Smart quotes
        text = (
            text.replace("bj", "4").replace("ij", "10").replace("tk", "8")
        )  # Custom replacements: BJ=4, IJ=10, TK=8
        return text

    @staticmethod
    def stem_text(text: str) -> str:
        """Stem text to root words for better matching"""
        text = TextProcessor.normalize(text)
        # Split into words, stem each, rejoin
        words = re.findall(r"\b\w+\b", text)
        stemmed = [TextProcessor.stemmer.stem(word) for word in words]
        return " ".join(stemmed)

    @staticmethod
    def extract_keywords(text: str) -> set:
        """Extract words from text (now uses stemming)"""
        text = TextProcessor.normalize(text)
        # Remove common words
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "of",
            "in",
            "is",
            "it",
            "to",
            "for",
        }
        words = set(re.findall(r"\b\w+\b", text)) - stopwords
        # Stem the words
        stemmed_words = {TextProcessor.stemmer.stem(word) for word in words}
        return stemmed_words

    @staticmethod
    def extract_pattern(text: str) -> Optional[Tuple[str, str]]:
        """Extract [Adjective] [Noun] pattern (now uses stemming)"""
        text = TextProcessor.normalize(text)
        # Pattern:  Capitalized words separated by space or dash
        parts = re.split(r"[\s\-]+", text)
        if len(parts) >= 2:
            # Stem both parts
            adj_stemmed = TextProcessor.stemmer.stem(parts[0])
            noun_stemmed = TextProcessor.stemmer.stem(" ".join(parts[1:]))
            return (adj_stemmed, noun_stemmed)
        return None


# ============================================================================
# 3. SIMILARITY MATCHING ENGINE
# ============================================================================


class SimilarityMatcher:
    """Handles various matching strategies"""

    # Thresholds for different strategies
    FUZZY_THRESHOLD = 0.80
    LEVENSHTEIN_THRESHOLD = 0.75
    KEYWORD_THRESHOLD = 0.5
    PATTERN_THRESHOLD = 0.65

    @staticmethod
    def exact_match(query: str, target: str) -> float:
        """Exact string matching"""
        norm_query = TextProcessor.normalize(query)
        norm_target = TextProcessor.normalize(target)
        return 1.0 if norm_query == norm_target else 0.0

    @staticmethod
    def fuzzy_match(query: str, target: str) -> float:
        """SequenceMatcher-based fuzzy matching"""
        norm_query = TextProcessor.normalize(query)
        norm_target = TextProcessor.normalize(target)

        if not norm_query or not norm_target:
            return 0.0

        ratio = SequenceMatcher(None, norm_query, norm_target).ratio()
        return ratio if ratio >= SimilarityMatcher.FUZZY_THRESHOLD else 0.0

    @staticmethod
    def levenshtein_match(query: str, target: str) -> float:
        """Levenshtein distance-based matching"""
        if not LEVENSHTEIN_AVAILABLE:
            return 0.0

        norm_query = TextProcessor.normalize(query)
        norm_target = TextProcessor.normalize(target)

        if not norm_query or not norm_target:
            return 0.0

        max_len = max(len(norm_query), len(norm_target))
        if max_len == 0:
            return 0.0

        distance = Levenshtein.distance(norm_query, norm_target)
        similarity = 1 - (distance / max_len)
        return (
            similarity if similarity >= SimilarityMatcher.LEVENSHTEIN_THRESHOLD else 0.0
        )

    @staticmethod
    def keyword_match(query: str, target: str) -> float:
        """Keyword overlap matching"""
        query_keywords = TextProcessor.extract_keywords(query)
        target_keywords = TextProcessor.extract_keywords(target)

        if not query_keywords or not target_keywords:
            return 0.0

        overlap = len(query_keywords & target_keywords)
        total = len(query_keywords | target_keywords)

        similarity = overlap / total if total > 0 else 0.0
        return similarity if similarity >= SimilarityMatcher.KEYWORD_THRESHOLD else 0.0

    @staticmethod
    def pattern_match(query: str, target: str) -> float:
        """Pattern-based matching (e.g., [Color] [Noun])"""
        query_pattern = TextProcessor.extract_pattern(query)
        target_pattern = TextProcessor.extract_pattern(target)

        if not query_pattern or not target_pattern:
            return 0.0

        # Score based on pattern part similarity
        adj_sim = SimilarityMatcher.fuzzy_match(query_pattern[0], target_pattern[0])
        noun_sim = SimilarityMatcher.fuzzy_match(query_pattern[1], target_pattern[1])

        # Both parts should be somewhat similar
        if adj_sim > 0.65 or noun_sim > 0.65:
            return (adj_sim + noun_sim) / 2

        return 0.0

    @staticmethod
    def semantic_match(query: str, target: str) -> Tuple[float, MatchStrategy]:
        """Semantic matching using multiple strategies"""
        scores = {
            MatchStrategy.EXACT: SimilarityMatcher.exact_match(query, target),
            MatchStrategy.FUZZY: SimilarityMatcher.fuzzy_match(query, target),
            MatchStrategy.KEYWORD: SimilarityMatcher.keyword_match(query, target),
            MatchStrategy.PATTERN: SimilarityMatcher.pattern_match(query, target),
        }

        if LEVENSHTEIN_AVAILABLE:
            scores[MatchStrategy.LEVENSHTEIN] = SimilarityMatcher.levenshtein_match(
                query, target
            )

        # Return highest score and corresponding strategy
        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]

        return best_score, best_strategy


# ============================================================================
# 4. SPREADSHEET ENGINE
# ============================================================================


class SpreadsheetEngine:
    """Main engine for spreadsheet analysis"""

    def __init__(self, data: Union[str, pd.DataFrame], file_type: str = "auto"):
        """
        Initialize the engine

        Args:
            data: File path or pandas DataFrame
            file_type: 'csv', 'excel', 'google_sheets', or 'auto'
        """
        self.df = self._load_data(data, file_type)
        self.logger = logger

        # Validate data
        if self.df is None or self.df.empty:
            raise ValueError("Failed to load or empty spreadsheet")

        self.logger.info(
            f"Loaded spreadsheet with {len(self.df)} rows and {len(self.df.columns)} columns"
        )

    def _load_data(
        self, data: Union[str, pd.DataFrame], file_type: str
    ) -> Optional[pd.DataFrame]:
        """Load data from various sources"""

        # If already a DataFrame, return it
        if isinstance(data, pd.DataFrame):
            return data

        if not isinstance(data, str):
            raise TypeError(f"Expected str or DataFrame, got {type(data)}")

        try:
            if file_type == "auto":
                if data.endswith(".csv"):
                    return pd.read_csv(data)
                elif data.endswith((".xlsx", ".xls")):
                    return pd.read_excel(data)
                else:
                    raise ValueError(f"Unknown file type for {data}")

            elif file_type == "csv":
                return pd.read_csv(data)

            elif file_type == "excel":
                return pd.read_excel(data)

            elif file_type == "google_sheets":
                # Convert Google Sheets URL to CSV export URL
                if "docs.google.com" in data:
                    sheet_id = re.search(r"/d/([a-zA-Z0-9-_]+)", data)
                    if sheet_id:
                        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id. group(1)}/export?format=csv"
                        return pd.read_csv(csv_url)
                raise ValueError("Invalid Google Sheets URL")

            else:
                raise ValueError(f"Unknown file_type: {file_type}")

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return None

    def search(
        self,
        query: str,
        search_column: Optional[str] = None,
        min_score: float = 0.50,
    ) -> List[MatchResult]:
        """
        Search for a value in the spreadsheet

        Args:
            query: The value to search for
            search_column:  Specific column to search in (None = all columns)
            min_score: Minimum similarity threshold (0-1). Default 0.50 (50%)

        Returns:
            List of MatchResult objects, sorted by score (highest first)
        """

        results = []

        # Determine which columns to search
        columns_to_search = [search_column] if search_column else self.df.columns

        for col in columns_to_search:
            if col not in self.df.columns:
                self.logger.warning(f"Column {col} not found in spreadsheet")
                continue

            for idx, value in self.df[col].items():
                if pd.isna(value) or value == "":
                    continue

                value_str = str(value).strip()

                # Skip suspiciously long entries (likely corrupt data)
                if len(value_str) > 500:
                    continue

                # Try semantic match
                score, strategy = SimilarityMatcher.semantic_match(query, value_str)

                # Only include results above the threshold
                if score >= min_score:
                    results.append(
                        MatchResult(
                            value=value_str,
                            column=col,
                            score=score,
                            strategy=strategy,
                            row_index=idx,
                        )
                    )

        # Sort by score (highest first)
        results.sort()
        return results  # Return ALL results above threshold

    def lookup(
        self, query: str, source_column: str, target_column: str, fuzzy: bool = True
    ) -> Optional[RelationalLookupResult]:
        """
        Relational lookup: Find a value in source_column, return from target_column

        Args:
            query: Value to find in source_column
            source_column: Column to search in
            target_column: Column to retrieve from
            fuzzy: Use fuzzy matching if exact match fails

        Returns:
            RelationalLookupResult or None
        """

        if source_column not in self.df.columns:
            self.logger.error(f"Source column {source_column} not found")
            return None

        if target_column not in self.df.columns:
            self.logger.error(f"Target column {target_column} not found")
            return None

        # Try exact match first
        for idx, value in self.df[source_column].items():
            if pd.isna(value):
                continue

            value_str = str(value).strip()

            if SimilarityMatcher.exact_match(query, value_str) > 0:
                target_value = self.df.loc[idx, target_column]
                return RelationalLookupResult(
                    source_column=source_column,
                    source_value=query,
                    target_column=target_column,
                    target_value=(
                        str(target_value) if not pd.isna(target_value) else "N/A"
                    ),
                    row_index=idx,
                )

        # Try fuzzy matching if enabled
        if fuzzy:
            best_match = None
            best_score = 0

            for idx, value in self.df[source_column].items():
                if pd.isna(value):
                    continue

                value_str = str(value).strip()
                score = SimilarityMatcher.fuzzy_match(query, value_str)

                if score > best_score:
                    best_score = score
                    best_match = (idx, value_str)

            if best_match and best_score > 0.80:
                idx, matched_value = best_match
                target_value = self.df.loc[idx, target_column]
                return RelationalLookupResult(
                    source_column=source_column,
                    source_value=matched_value,
                    target_column=target_column,
                    target_value=(
                        str(target_value) if not pd.isna(target_value) else "N/A"
                    ),
                    row_index=idx,
                )

        return None

    def batch_lookup(
        self,
        queries: List[str],
        source_column: str,
        target_column: str,
        fuzzy: bool = True,
    ) -> List[Optional[RelationalLookupResult]]:
        """
        Batch relational lookups

        Args:
            queries: List of values to find
            source_column: Column to search in
            target_column: Column to retrieve from
            fuzzy: Use fuzzy matching

        Returns:
            List of RelationalLookupResult objects
        """
        return [self.lookup(q, source_column, target_column, fuzzy) for q in queries]

    def get_column_info(self) -> Dict[str, Dict]:
        """Get information about all columns"""
        info = {}
        for col in self.df.columns:
            non_null = self.df[col].notna().sum()
            info[col] = {
                "total_rows": len(self.df),
                "non_null": non_null,
                "null_count": len(self.df) - non_null,
                "unique_values": self.df[col].nunique(),
                "data_type": str(self.df[col].dtype),
            }
        return info

    def validate_data_quality(self) -> Dict[str, any]:
        """Check data quality and return report"""
        report = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "missing_data": self.df.isna().sum().to_dict(),
            "duplicates": int(self.df.duplicated().sum()),
            "warnings": [],
        }

        # Check for suspicious data
        for col in self.df.columns:
            for idx, val in self.df[col].items():
                if isinstance(val, str) and len(val) > 500:
                    report["warnings"].append(
                        f"Row {idx}, Column '{col}': Suspiciously long entry ({len(val)} chars) - may be corrupt"
                    )

        return report

    def get_row_data(self, row_index: int) -> Dict:
        """Get all data from a specific row"""
        if row_index < 0 or row_index >= len(self.df):
            return {}
        return self.df.iloc[row_index].to_dict()

    def get_safe_value(
        self, row_data: Dict, column_variants: List[str], default: str = "N/A"
    ) -> str:
        """
        Safely get a value from row_data, trying multiple column name variants.

        Args:
            row_data: The row dictionary
            column_variants: List of possible column names to try
            default: Default value if none found

        Returns:
            The value if found, otherwise default
        """
        for col in column_variants:
            if col in row_data and row_data[col] not in [None, "", "nan"]:
                return str(row_data[col])
        return default
