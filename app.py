import streamlit as st
from google import genai as google_genai
from groq import Groq
from huggingface_hub import InferenceClient
from pinecone import Pinecone
import re
import json
from datetime import datetime, timedelta
import logging
import sys
import os
import traceback
import io
from dateutil import parser as dateutil_parser
import pandas as pd
import numpy as np
from spreadsheet_engine import SpreadsheetEngine

# Google Drive API imports
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.api_core.exceptions import GoogleAPICallError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import PyPDF2

# Configure logging (no UI output)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("drive_service_debug.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# --- 1.CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="'Remcensus",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR CONSENSUS-STYLE UI ---
st.markdown("""
<style>
    /* Modern, clean layout */
    .main .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Clean sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] .element-container {
        padding: 0.5rem 0;
    }
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Modern heading styling */
    h1, h2, h3 {
        font-weight: 600;
        color: #1a1a1a;
    }
    
    /* Branding section */
    .remcensus-brand {
        text-align: center;
        margin: 2rem 0 1rem 0;
    }
    
    .remcensus-title {
        font-size: 3rem;
        font-weight: 700;
        color: #16a8b6;
        margin-bottom: 0.5rem;
    }
    
    .remcensus-tagline {
        font-size: 1.25rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Search bar styling */
    .stTextInput > div > div > input {
        border-radius: 50px;
        border: 2px solid #e0e0e0;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #16a8b6;
        box-shadow: 0 0 0 3px rgba(22, 168, 182, 0.1), 0 4px 12px rgba(0,0,0,0.08);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #999;
    }
    
    /* Action buttons styling */
    .action-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .action-button {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        color: #333;
        text-align: center;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
    }
    
    .action-button:hover {
        background-color: #16a8b6;
        color: white;
        border-color: #16a8b6;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(22, 168, 182, 0.2);
    }
    
    /* Card-style results */
    .result-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Improved spacing */
    .stMarkdown {
        line-height: 1.6;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #16a8b6;
    }
    
    /* Sidebar navigation styling */
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #16a8b6;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        font-size: 0.95rem;
    }
    
    /* Sidebar navigation links */
    [data-testid="stSidebar"] .stMarkdown p {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        cursor: pointer;
        margin: 0.25rem 0;
    }
    
    [data-testid="stSidebar"] .stMarkdown p:hover {
        background-color: rgba(22, 168, 182, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        border: 2px solid #16a8b6;
        background-color: #16a8b6;
        color: white;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1293a0;
        border-color: #1293a0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(22, 168, 182, 0.3);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #16a8b6;
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 8px;
        background-color: #f8f9fa;
        font-weight: 500;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #16a8b6;
    }
    
    /* Clean dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Feature highlights */
    .feature-highlight {
        padding: 1.5rem;
        border-radius: 12px;
        background: #f8f9fa;
        border: 1px solid #e8e9ea;
        transition: all 0.3s ease;
    }
    
    .feature-highlight:hover {
        background: #ffffff;
        border-color: #16a8b6;
        box-shadow: 0 4px 12px rgba(22, 168, 182, 0.1);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #16a8b6 !important;
    }
    
    /* Success/info messages */
    .stSuccess {
        background-color: rgba(22, 168, 182, 0.1);
        border-radius: 8px;
        border-left: 4px solid #16a8b6;
    }
    
    /* Warning messages */
    .stWarning {
        border-radius: 8px;
        border-left: 4px solid #ffa726;
    }
    
    /* Error messages */
    .stError {
        border-radius: 8px;
        border-left: 4px solid #ef5350;
    }
</style>
""", unsafe_allow_html=True)

# --- SCORE THRESHOLD CONFIGURATION ---
SCORE_THRESHOLD = 8  # Minimum score for a file to be included in results

# --- FOLDER STRUCTURE MAP ---
RACRL_FOLDER_MAP = {
    "Education": {
        "folder_id": "A. 'Remducation",
        "keywords": ["education", "curriculum", "learning", "training", "beginner"],
        "priority": 1,
    },
    "Registry": {
        "folder_id": "B. The Registry",
        "keywords": ["registry", "register", "national"],
        "priority": 2,
    },
    "Puzzles": {
        "folder_id": "C. 'uzzles",
        "keywords": ["puzzle", "puzzles", "challenge", "scenario"],
        "priority": 3,
    },
    "Exam Prep": {
        "folder_id": "D. Exam 'rep",
        "keywords": ["exam", "prep", "preparation", "curriculum", "variation"],
        "priority": 4,
    },
    "Publications": {
        "folder_id": "E. 'ublications",
        "keywords": ["publication", "research", "paper", "article", "mash"],
        "priority": 5,
    },
    "SPUDS": {
        "folder_id": "F. SPUDS",
        "keywords": ["spuds", "guide", "hymn", "trial"],
        "priority": 6,
    },
    "RANZCCP": {
        "folder_id": "G. RANZCCP",
        "keywords": ["ranzccp", "establishment"],
        "priority": 7,
    },
    "Minutes": {
        "folder_id": "H. Minutes",
        "keywords": ["minutes", "meeting", "nha", "vha", "waha"],
        "priority": 8,
        "subcategories": {
            "NHA": {"keywords": ["nha", "national high able"], "priority": 1},
            "VHA": {"keywords": ["vha", "victorian high able"], "priority": 2},
            "WAHA": {"keywords": ["waha", "western"], "priority": 3},
        },
    },
    "Rulings": {
        "folder_id": "I. Other Rulings",
        "keywords": ["ruling", "recommendation", "deck chair"],
        "priority": 9,
    },
    "History": {
        "folder_id": "J. Extra History",
        "keywords": ["history", "anecdote", "event", "botsquali", "sahasa", "game"],
        "priority": 10,
    },
}

# Year patterns (for sorting - most recent first)
YEAR_PRIORITY = {
    "2025": 1,
    "202BJ": 2,
    "2023": 3,
    "2022": 4,
    "2021": 5,
    "2020": 6,
    "2017": 7,
    "2016": 8,
    "2015": 9,
    "2010": 10,
    "pre-2015": 11,
    "old": 12,
}


# --- DATE/TIME PARSING & FILTERING ---
def parse_date(date_string):
    """
    Parse various date formats intelligently.
    Supports:  DMY, MDY, YMD, ISO 8601, Full, Medium, etc.

    Returns:  datetime object or None if parsing fails
    """
    if not date_string:
        return None

    try:
        # Try dateutil parser first (most flexible)
        # dayfirst=True prioritizes DMY over MDY
        parsed_date = dateutil_parser.parse(date_string, dayfirst=True, fuzzy=True)
        logger.debug(
            f"Successfully parsed date: '{date_string}' -> {parsed_date.isoformat()}"
        )
        return parsed_date
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse date '{date_string}': {str(e)}")
        return None


def extract_date_constraints_from_query(query):
    """
    Extract date-based constraints from query.
    Detects: 'most recent', 'latest', 'within X days/months/years', 'after/before DATE', 'in YEAR', etc.

    Returns: dict with constraint info
    """
    query_lower = query.lower()
    constraints = {
        "most_recent": False,
        "latest": False,
        "within_days": None,
        "within_months": None,
        "within_years": None,
        "after_date": None,
        "before_date": None,
        "specific_year": None,
    }

    # Check for "most recent" / "latest"
    if "most recent" in query_lower or "latest" in query_lower:
        constraints["most_recent"] = True
        constraints["latest"] = True
        logger.debug("Detected 'most recent' constraint")

    # Check for "within X days"
    within_days_match = re.search(r"within\s+(\d+)\s+days?", query_lower)
    if within_days_match:
        constraints["within_days"] = int(within_days_match.group(1))
        logger.debug(f"Detected 'within {constraints['within_days']} days' constraint")

    # Check for "within X months"
    within_months_match = re.search(r"within\s+(\d+)\s+months?", query_lower)
    if within_months_match:
        constraints["within_months"] = int(within_months_match.group(1))
        logger.debug(
            f"Detected 'within {constraints['within_months']} months' constraint"
        )

    # Check for "within X years"
    within_years_match = re.search(
        r"within\s+(?:the\s+)?last\s+(\d+)\s+years?", query_lower
    )
    if within_years_match:
        constraints["within_years"] = int(within_years_match.group(1))
        logger.debug(
            f"Detected 'within {constraints['within_years']} years' constraint"
        )

    # Check for "after DATE"
    after_match = re.search(r"after\s+([^\s,]+(?:\s+[^\s,]+)*)", query_lower)
    if after_match:
        date_str = after_match.group(1)
        parsed = parse_date(date_str)
        if parsed:
            constraints["after_date"] = parsed
            logger.debug(f"Detected 'after {parsed.isoformat()}' constraint")

    # Check for "before DATE"
    before_match = re.search(r"before\s+([^\s,]+(?:\s+[^\s,]+)*)", query_lower)
    if before_match:
        date_str = before_match.group(1)
        parsed = parse_date(date_str)
        if parsed:
            constraints["before_date"] = parsed
            logger.debug(f"Detected 'before {parsed.isoformat()}' constraint")

    # Check for specific year (e.g., "in 2025", "from 2020")
    year_match = re.search(r"(?:in|from)\s+(20\d{2}|19\d{2})", query_lower)
    if year_match:
        constraints["specific_year"] = int(year_match.group(1))
        logger.debug(
            f"Detected 'specific year {constraints['specific_year']}' constraint"
        )

    return constraints


def apply_date_filter(files, constraints, current_time=None):
    """
    Filter files based on date constraints.

    Args:
        files: List of file objects with 'modifiedTime' or 'createdTime'
        constraints: Dict from extract_date_constraints_from_query
        current_time: Reference time for relative calculations (defaults to now)

    Returns: Filtered list of files
    """
    if current_time is None:
        current_time = datetime.now(
            tz=datetime.now().astimezone().tzinfo or __import__("datetime").timezone.utc
        )

    if not any(constraints.values()):
        logger.debug("No date constraints detected, returning all files")
        return files

    filtered_files = []

    for file in files:
        # Get the file's modified or created time
        time_str = file.get("modifiedTime") or file.get("createdTime")
        if not time_str:
            logger.warning(
                f"File {file.get('name')} has no time metadata, including anyway"
            )
            filtered_files.append(file)
            continue

        try:
            file_time = dateutil_parser.isoparse(time_str)

            # Make sure both datetimes are timezone-aware for comparison
            if file_time.tzinfo is None:
                file_time = file_time.replace(
                    tzinfo=datetime.now().astimezone().tzinfo
                    or __import__("datetime").timezone.utc
                )
            if current_time.tzinfo is None:
                current_time = current_time.replace(
                    tzinfo=datetime.now().astimezone().tzinfo
                    or __import__("datetime").timezone.utc
                )

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse file time {time_str}: {str(e)}")
            filtered_files.append(file)  # Include if parsing fails
            continue

        include_file = True

        # Check "within X days" constraint
        if constraints["within_days"]:
            cutoff = current_time - timedelta(days=constraints["within_days"])
            if file_time < cutoff:
                include_file = False
                logger.debug(
                    f"File {file.get('name')} excluded (older than {constraints['within_days']} days)"
                )

        # Check "within X months" constraint
        if constraints["within_months"] and include_file:
            from dateutil.relativedelta import relativedelta

            cutoff = current_time - relativedelta(months=constraints["within_months"])
            if file_time < cutoff:
                include_file = False
                logger.debug(
                    f"File {file.get('name')} excluded (older than {constraints['within_months']} months)"
                )

        # Check "within X years" constraint
        if constraints["within_years"] and include_file:
            cutoff = current_time - timedelta(
                days=constraints["within_years"] * 365
            )  # Approximate
            if file_time < cutoff:
                include_file = False
                logger.debug(
                    f"File {file.get('name')} excluded (older than {constraints['within_years']} years)"
                )

        # Check "after DATE" constraint
        if constraints["after_date"] and include_file:
            if file_time < constraints["after_date"]:
                include_file = False
                logger.debug(
                    f"File {file.get('name')} excluded (before {constraints['after_date'].isoformat()})"
                )

        # Check "before DATE" constraint
        if constraints["before_date"] and include_file:
            if file_time > constraints["before_date"]:
                include_file = False
                logger.debug(
                    f"File {file.get('name')} excluded (after {constraints['before_date'].isoformat()})"
                )

        # Check specific year constraint
        if constraints["specific_year"] and include_file:
            if file_time.year != constraints["specific_year"]:
                include_file = False
                logger.debug(
                    f"File {file.get('name')} excluded (not from year {constraints['specific_year']})"
                )

        if include_file:
            filtered_files.append(file)

    logger.info(
        f"Date filtering:  {len(files)} files -> {len(filtered_files)} files after constraints"
    )
    return filtered_files


# --- GOOGLE DRIVE SERVICE INITIALIZATION ---
def initialize_drive_service():
    """Initialize Google Drive service with detailed error logging."""
    try:
        logger.info("Starting Google Drive Service Initialization")

        try:
            cred_json = st.secrets.get("GDRIVE_SERVICE_ACCOUNT_JSON")
        except Exception as e:
            logger.error(f"Failed to access Streamlit secrets: {str(e)}")
            cred_json = None

        if not cred_json:
            logger.error("GDRIVE_SERVICE_ACCOUNT_JSON not found in Streamlit secrets")
            return None

        logger.info("GDRIVE_SERVICE_ACCOUNT_JSON found in Streamlit secrets")

        if isinstance(cred_json, dict):
            cred_content = cred_json
        elif isinstance(cred_json, str):
            cred_content = json.loads(cred_json)
        else:
            logger.error(f"Unexpected credentials type: {type(cred_json)}")
            return None

        required_fields = [
            "type",
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
        ]
        missing_fields = [
            field for field in required_fields if field not in cred_content
        ]

        if missing_fields:
            logger.error(f"Missing required fields in credentials: {missing_fields}")
            return None

        logger.info("Creating service account credentials...")
        try:
            scopes = [
                "https://www.googleapis.com/auth/drive.metadata.readonly",
                "https://www.googleapis.com/auth/drive.readonly",
            ]

            credentials = Credentials.from_service_account_info(
                cred_content, scopes=scopes
            )
            logger.info("Service account credentials created successfully")

        except ValueError as e:
            logger.error(f"Invalid service account credentials: {str(e)}")
            return None

        try:
            if not credentials.valid:
                logger.debug("Credentials not valid, attempting refresh...")
                credentials.refresh(Request())
                logger.info("Credentials refreshed successfully")
        except Exception as e:
            logger.error(f"Failed to refresh credentials:  {str(e)}")
            logger.debug("Continuing with unrefreshed credentials...")

        logger.info("Building Google Drive service...")
        try:
            service = build(
                "drive", "v3", credentials=credentials, cache_discovery=False
            )
            logger.info("Google Drive service built successfully")

        except HttpError as e:
            logger.error(f"HTTP error building Drive service: {e.resp.status}")
            return None
        except GoogleAPICallError as e:
            logger.error(f"Google API error building Drive service: {str(e)}")
            return None

        logger.info("Testing Google Drive service...")
        try:
            results = (
                service.files()
                .list(spaces="drive", fields="files(id, name)", pageSize=1)
                .execute()
            )
            logger.info("Google Drive service test successful")

        except HttpError as e:
            logger.error(f"HTTP error testing Drive service: {e.resp.status}")
            return None

        logger.info("Google Drive Service Initialization COMPLETED SUCCESSFULLY")
        return service

    except Exception as e:
        logger.critical(
            f"Unexpected error in initialize_drive_service: {type(e).__name__} - {str(e)}",
            exc_info=True,
        )
        return None


# --- CONTENT EXTRACTION HELPER ---
def extract_file_content(drive_service, file_id, mime_type, file_name):
    """Extract content from Google Drive files (Google Docs, PDFs, Google Sheets)."""
    try:
        logger.info(f"Extracting content from:   {file_name} (MIME: {mime_type})")

        if "vnd.google-apps.document" in mime_type:
            logger.debug("Detected Google Docs format")
            try:
                request = drive_service.files().export(
                    fileId=file_id, mimeType="text/plain"
                )
                content = request.execute().decode("utf-8")
                logger.info(
                    f"Successfully extracted Google Docs content ({len(content)} chars)"
                )
                return content, True, None
            except Exception as e:
                logger.error(f"Failed to extract Google Docs:   {str(e)}")
                return "", False, f"Google Docs extraction failed: {str(e)}"

        elif "vnd.google-apps.spreadsheet" in mime_type:
            logger.debug("Detected Google Sheets format")
            try:
                request = drive_service.files().export(
                    fileId=file_id, mimeType="text/csv"
                )
                content = request.execute().decode("utf-8")
                logger.info(
                    f"Successfully extracted Google Sheets content ({len(content)} chars)"
                )
                return content, True, None
            except Exception as e:
                logger.error(f"Failed to extract Google Sheets:  {str(e)}")
                return "", False, f"Google Sheets extraction failed: {str(e)}"

        elif "application/pdf" in mime_type:
            logger.debug("Detected PDF format")
            try:
                request = drive_service.files().get_media(fileId=file_id)
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False

                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.debug(
                            f"Download progress: {int(status.progress() * 100)}%"
                        )

                file_stream.seek(0)

                pdf_reader = PyPDF2.PdfReader(file_stream)
                content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_content = page.extract_text()
                        content += f"\n--- Page {page_num + 1} ---\n{page_content}"
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract page {page_num + 1}: {str(e)}"
                        )
                        content += f"\n--- Page {page_num + 1} ---\n[Unable to extract text from this page]"

                logger.info(
                    f"Successfully extracted PDF content ({len(content)} chars from {len(pdf_reader.pages)} pages)"
                )
                return content, True, None
            except Exception as e:
                logger.error(f"Failed to extract PDF:  {str(e)}")
                return "", False, f"PDF extraction failed: {str(e)}"

        elif "vnd.openxmlformats-officedocument.wordprocessingml.document" in mime_type:
            logger.debug("Detected DOCX format")
            try:
                request = drive_service.files().get_media(fileId=file_id)
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False

                while not done:
                    status, done = downloader.next_chunk()

                file_stream.seek(0)

                try:
                    from docx import Document

                    doc = Document(file_stream)
                    content = "\n".join([para.text for para in doc.paragraphs])
                    logger.info(
                        f"Successfully extracted DOCX content ({len(content)} chars)"
                    )
                    return content, True, None
                except ImportError:
                    logger.warning(
                        "python-docx not available, returning generic message"
                    )
                    return (
                        "[DOCX file content - unable to extract without python-docx library]",
                        True,
                        "DOCX extraction limited",
                    )
            except Exception as e:
                logger.error(f"Failed to extract DOCX:  {str(e)}")
                return "", False, f"DOCX extraction failed: {str(e)}"

        else:
            logger.warning(f"Unsupported MIME type: {mime_type}")
            return "", False, f"Unsupported file format: {mime_type}"

    except Exception as e:
        logger.error(
            f"Unexpected error extracting file content: {type(e).__name__} - {str(e)}"
        )
        return "", False, f"Unexpected error:  {str(e)}"

# Extract content setting (default:  enabled)
extract_content = True


def fetch_all_drive_files_cached(drive_service, drive_id):
    """
    Fetch ALL files from Google Drive recursively and cache them.
    This function is called once at startup to avoid repeated API calls.
    """
    try:
        logger.info(f"[CACHE] Fetching all files from drive/folder {drive_id}")

        all_items = []
        folders_to_search = [drive_id]
        searched_folders = set()
        folder_count = 0

        logger.info("[CACHE] Starting recursive folder traversal...")
        while folders_to_search:
            current_folder = folders_to_search.pop(0)

            if current_folder in searched_folders:
                continue

            searched_folders.add(current_folder)
            folder_count += 1
            logger.info(f"[CACHE] [FOLDER #{folder_count}] Searching: {current_folder}")

            try:
                query_string = f"'{current_folder}' in parents and trashed=false"
                resp = (
                    drive_service.files()
                    .list(
                        q=query_string,
                        spaces="drive",
                        pageSize=100,
                        fields="files(id,name,createdTime,modifiedTime,owners(displayName,emailAddress),mimeType,webViewLink,size,description)",
                    )
                    .execute()
                )

                items = resp.get("files", [])
                logger.info(f"[CACHE]   -> Found {len(items)} items")

                for item in items:
                    name = item.get("name", "UNKNOWN")
                    mime = item.get("mimeType", "UNKNOWN")
                    is_folder = mime == "application/vnd.google-apps.folder"

                    if is_folder:
                        logger.info(f"[CACHE]     [FOLDER] {name}")
                        folders_to_search.append(item["id"])
                    else:
                        logger.info(f"[CACHE]     [FILE] {name}")
                        all_items.append(item)

            except HttpError as e:
                logger.warning(f"[CACHE] Error searching folder {current_folder}: {e}")
                continue

        logger.info(f"[CACHE] Complete: Found {len(all_items)} files total")

        # Normalize datetime strings
        for item in all_items:
            if "createdTime" in item:
                try:
                    dt = datetime.fromisoformat(
                        item["createdTime"].replace("Z", "+00:00")
                    )
                    item["createdTimeISO"] = dt.isoformat()
                except Exception as dt_e:
                    logger.warning(
                        f"Failed to parse datetime {item['createdTime']}: {dt_e}"
                    )
                    item["createdTimeISO"] = item.get("createdTime")

            if "modifiedTime" in item:
                try:
                    dt = datetime.fromisoformat(
                        item["modifiedTime"].replace("Z", "+00:00")
                    )
                    item["modifiedTimeISO"] = dt.isoformat()
                except Exception as dt_e:
                    logger.warning(
                        f"Failed to parse datetime {item['modifiedTime']}: {dt_e}"
                    )
                    item["modifiedTimeISO"] = item.get("modifiedTime")

        # Sort by modified time
        all_items.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)

        return all_items

    except Exception as e:
        logger.error(
            f"[CACHE] Unexpected error fetching drive items: {type(e).__name__} - {str(e)}",
            exc_info=True,
        )
        return []


# SECURE CONNECTION
if "init_done" not in st.session_state:
    try:
        logger.info("Initializing Pinecone and AI clients...")
        pc = Pinecone(api_key=st.secrets["PINECONE_KEY"])
        st.session_state.pc_index = pc.Index(st.secrets["PINECONE_INDEX"])
        st.session_state.google_client = google_genai.Client(
            api_key=st.secrets["GEMINI_KEY"]
        )
        st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        st.session_state.hf_client = InferenceClient(
            api_key=st.secrets["HUGGINGFACE_KEY"]
        )
        logger.info("Pinecone and AI clients initialized successfully")

        logger.info("Initializing Google Drive service...")
        st.session_state.drive_service = initialize_drive_service()

        if st.session_state.drive_service:
            logger.info("Drive service initialized successfully")
            
            # Cache all Drive files at startup to avoid repeated API calls
            logger.info("Caching Google Drive files...")
            with st.spinner("📥 Loading Google Drive files into cache..."):
                st.session_state.drive_files_cache = fetch_all_drive_files_cached(
                    st.session_state.drive_service,
                    "10B8EsEQ2TlzQP5ADD43TcDSs_xp3plj9"  # Main drive folder ID
                )
            logger.info(f"✅ Cached {len(st.session_state.drive_files_cache)} files from Drive")
        else:
            logger.warning("Drive service initialization returned None")
            st.session_state.drive_files_cache = []

        st.session_state.init_done = True
        logger.info("Session initialization completed")

    except Exception as e:
        logger.error(f"Security/Initialization Error: {e}", exc_info=True)
        st.error(f"🔐 Security Error: {e}")
        st.stop()

# Initialize navigation state
if "current_view" not in st.session_state:
    st.session_state.current_view = "home"

# Developer mode (can be toggled via environment variable or config)
dev_mode = False  # Set to True to enable developer debugging features


# --- 2.LINGUISTIC TRANSFORMATION ENGINE ---
def enforce_rem_lexicon(text):
    text = re.sub(r"\bP", "'", text)
    text = re.sub(r"\bp", "'", text)
    replacements = {
        "Table": "'able",
        "table": "'able",
        "Drink": "'rink",
        "drink": "'rink",
    }
    for word, replacement in replacements.items():
        text = text.replace(word, replacement)
    num_map = [
        (r"\b444\b", "BJBJBJ"),
        (r"\b888\b", "TKTKTK"),
        (r"\b400\b", "BJ00"),
        (r"\b800\b", "TK00"),
        (r"\b40\b", "BJ0"),
        (r"\b80\b", "TK0"),
        (r"\b14\b", "1BJ"),
        (r"\b18\b", "1TK"),
        (r"\b4\b", "BJ"),
        (r"\bfour\b", "bon jovi"),
        (r"\b8\b", "TK"),
        (r"\beight\b", "takahashi"),
        (r"\b10\b", "IJ"),
        (r"\bten\b", "iku jo"),
    ]
    for pattern, sub in num_map:
        text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    return text


# --- 3.UNIVERSAL SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "You are a neutral information retrieval system for the 'Remier League archives.\n\n"
    "Provide factual, detailed, and comprehensive answers based on the context provided.\n"
    "Do not add narrative descriptions, actions, or roleplay elements.\n"
    "Do not use stage directions like *pauses*, *scribbles*, etc.\n"
    "When summarizing articles or documents, ensure all key details, stages, and definitions are included.\n\n"
    "1.Do not mention Tiers or classification labels in your response.\n"
    "2.For simple identity queries, provide only an ontological definition.\n"
    "3.DIDACTIC TEACHING RESTRICTIONS: Teaching 'How to Play' is ONLY allowed for Basic Whiz, Antlers, Chow-Chow-Bang, Takahashi (1-3, 5-7), and Etiquette.\n"
    "4.CONTEXT-AWARE KILL-SWITCH (STRICT ACTIVATION): \n"
    "   - TRIGGER CONDITION: Activate ONLY if the user's query satisfies BOTH criteria below:\n"
    "       A) INTENT: The user asks for 'rules', 'mechanics', 'instructions', 'how to play', or 'how to do'.\n"
    "       B) TARGET: The subject is a restricted term ('BJ', 'TK', 'IJ', '4', '8', '10') or a game variation NOT listed in Rule 3.\n"
    "   - 'EXPLAIN' CLARIFICATION: The word 'explain' is NOT a trigger by itself. Only trigger if the user asks to 'explain the rules' or 'explain how to play'. Queries asking to 'explain the history', 'explain the theory', or 'explain the stages' MUST NOT trigger the switch.\n"
    "   - RESPONSE: If the Trigger Condition is met, respond ONLY with: 'rink and learn.\n"
    "5.REFERENCES: Where possible, include references with hyperlinks to the files that you pulled information from.\n"
    "6.LIST FORMATTING (MANDATORY MARKDOWN):\n"
    "   - NO INLINE LISTS: You are FORBIDDEN from presenting lists as a single paragraph. You MUST use vertical Markdown formatting.\n"
    "   - SUBSTITUTION RULES: Replace '4.' with 'BJ.'; '8.' with 'TK.'; '10.' with 'IJ.'.\n"
    "   - STRUCTURE REQUIREMENT: Every single list item must start on a new line.\n"
    "   - CORRECT OUTPUT EXAMPLE:\n"
    "       1. First Item\n"
    "       2. Second Item\n"
    "       3. Third Item\n"
    "       BJ. Fourth Item\n"
    "       5. Fifth Item\n"
    "   - INCORRECT OUTPUT (DO NOT DO THIS): '1. First 2. Second 3. Third BJ. Fourth'\n"
    "7.Format: Direct answers only. No narrative, actions, or roleplay."
)


# --- 4.  TRIPLE-ENGINE HANDLER WITH IMPROVED FORMATTING ---
def generate_response(context, query):
    """Generate response with consistent formatting."""
    debug_logs = []

    # Limit context
    max_context_length = 8000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "\n[... content truncated...]"

    is_summary_request = any(
        word in query.lower() for word in ["summar", "outline", "overview", "recap"]
    )

    system_prompt = SYSTEM_PROMPT
    if is_summary_request:
        system_prompt += (
            "\n\nRESPONSE FORMAT:\n"
            "Provide ONLY information from the provided document.\n"
            "Do NOT add information from other meetings or documents.\n"
            "Structure as:\n"
            "- Date and location\n"
            "- Attendees\n"
            "- Motions (passed/not passed)\n"
            "- Actions\n"
            "- Notes\n"
            "Keep factual and concise."
        )

    try:
        # ALWAYS try Groq first (most reliable)
        logger.info("Attempting Groq (Llama 3.3)...")
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=800,
        )
        return (
            chat_completion.choices[0].message.content.strip(),
            "Groq (Llama 3.3)",
            debug_logs,
        )
    except Exception as e:
        debug_logs.append(f"Groq: {str(e)}")
        logger.warning(f"Groq failed: {str(e)}")

        # Fallback to Gemini
        try:
            logger.info("Attempting Gemini...")
            response = st.session_state.google_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Context:\n{context}\n\nQuery: {query}",
                config={"system_instruction": system_prompt},
            )
            return response.text.strip(), "Gemini (1.5 Flash)", debug_logs
        except Exception as e_gem:
            debug_logs.append(f"Gemini: {str(e_gem)}")
            logger.warning(f"Gemini failed: {str(e_gem)}")

            # Only use HF as last resort
            try:
                logger.info("Attempting HuggingFace...")
                response = st.session_state.hf_client.chat_completion(
                    model="meta-llama/Llama-3.2-70B-Instruct",  # Use 70B not 3B
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuery: {query}",
                        },
                    ],
                    max_tokens=800,
                    temperature=0.5,
                )
                return (
                    response.choices[0].message.content.strip(),
                    "HuggingFace (Llama 3.2 70B)",
                    debug_logs,
                )
            except Exception as e_hf:
                logger.error(f"All engines failed")
                return "⚠️ Unable to generate summary", "OFFLINE", debug_logs


# --- 5.FOLDER STRUCTURE INTELLIGENCE ---
def match_category(query_lower, folder_map):
    """Intelligently match query to folder categories."""
    best_match = (None, None, 0)

    for category, info in folder_map.items():
        if "subcategories" in info:
            for subcat_name, subcat_info in info["subcategories"].items():
                subcat_keywords = subcat_info.get("keywords", [])
                for keyword in subcat_keywords:
                    if keyword in query_lower:
                        confidence = len(keyword) / len(query_lower)
                        if confidence > best_match[2]:
                            best_match = (category, subcat_name, confidence)

    if best_match[2] == 0:
        for category, info in folder_map.items():
            category_keywords = info.get("keywords", [])
            for keyword in category_keywords:
                if keyword in query_lower:
                    confidence = len(keyword) / len(query_lower)
                    if confidence > best_match[2]:
                        best_match = (category, None, confidence)

    return best_match


# --- 6.GOOGLE DRIVE SEARCH WITH IMPROVED SCORING & DATE FILTERING ---

def fetch_drive_recent_files(
    drive_id, search_query=None, score_threshold=SCORE_THRESHOLD
):
    """
    Fetch files with intelligent scoring based on filename matches and date filtering.
    Now uses cached file list instead of making API calls.
    """
    try:
        logger.info(f"Searching cached files with query: {search_query}")

        # Use cached files instead of fetching from API
        if "drive_files_cache" not in st.session_state:
            logger.warning("Drive cache not available, falling back to live fetch")
            # Fallback to live fetch if cache is not available
            drive_service = st.session_state.get("drive_service")
            if drive_service is None:
                logger.error("Drive service is None in fetch_drive_recent_files")
                raise RuntimeError("Drive service not initialized.")
            all_items = fetch_all_drive_files_cached(drive_service, drive_id)
        else:
            logger.info(f"Using cached files: {len(st.session_state.drive_files_cache)} files available")
            all_items = st.session_state.drive_files_cache.copy()

        # Extract date constraints from query
        date_constraints = None
        if search_query:
            date_constraints = extract_date_constraints_from_query(search_query)
            logger.info(f"Date constraints extracted: {date_constraints}")

        # Apply date filtering
        if date_constraints and any(date_constraints.values()):
            all_items = apply_date_filter(all_items, date_constraints)
            logger.info(f"After date filtering: {len(all_items)} files")

        # Detect category
        category_match, subcategory_match, category_confidence = None, None, 0
        if search_query:
            category_match, subcategory_match, category_confidence = match_category(
                search_query.lower(), RACRL_FOLDER_MAP
            )
            logger.info(
                f"Query intent detected: Category={category_match}, Subcategory={subcategory_match}"
            )

        # IMPROVED SCORING (v21)
        if search_query:
            search_terms = [
                term for term in search_query.lower().split() if len(term) > 2
            ]
            scored_items = []

            # [INSERT PART 1 HERE: Define the map and find matches]
            # Manual Keyword Overrides (Patch for specific documents)
            keyword_map = {
                "lily": ["REMIER_Trial"],  # If query has 'lily', boost 'REMIER_Trial'
                "schmeer": ["REMIER_Trial"],
                "gravity": ["Gravity_Well"],
                "cpc": ["Spartan_Supremacy"],
                "boat": ["Spartan_Supremacy"],
            }
            
            # Check for keywords in query
            boost_files = []
            if search_query:
                for keyword, target_files in keyword_map.items():
                    if keyword in search_query.lower():
                        boost_files.extend(target_files)
            # [END PART 1]
            
            logger.info(
                f"Scoring {len(all_items)} files with improved filename-first strategy"
            )

            for item in all_items:
                name_lower = item.get("name", "").lower()
               
                # [INSERT PART 2 HERE: Apply the boost inside the loop]
                # Apply Manual Boost
                for boost_name in boost_files:
                    if boost_name.lower() in name_lower:
                        # Massive boost to ensure it's picked
                        # We use a temporary variable so we don't mess up the logic below
                        pass 
                
                desc_lower = (
                    item.get("description", "").lower()
                    if item.get("description")
                    else ""
                )
                score = 0
                
                # [ACTUAL BOOST APPLICATION]
                # Add this right here, before the other checks:
                for boost_name in boost_files:
                    if boost_name.lower() in name_lower:
                        score += 100

                # **MASSIVE BOOST FOR FILENAME MATCHES** (v21 improvement)
                for term in search_terms:
                    if term in name_lower:
                        score += 20  # Increased from 3 to 20 for filename matches

                # Small boost for description matches
                for term in search_terms:
                    if term in desc_lower:
                        score += 1

                # Category matching
                if category_match:
                    category_info = RACRL_FOLDER_MAP.get(category_match, {})
                    if any(
                        kw in name_lower for kw in category_info.get("keywords", [])
                    ):
                        score += 5

                    if subcategory_match and "subcategories" in category_info:
                        subcat_info = category_info["subcategories"].get(
                            subcategory_match, {}
                        )
                        if any(
                            kw in name_lower for kw in subcat_info.get("keywords", [])
                        ):
                            score += 50

                if score > 0:
                    scored_items.append((score, item))
                    logger.debug(f"File:   {item.get('name')} - Score: {score}")

            # Sort by score, then by recency
            def get_modified_timestamp(item):
                try:
                    dt = datetime.fromisoformat(
                        item.get("modifiedTime", "1970-01-01").replace("Z", "+00:00")
                    )
                    return dt.timestamp()
                except Exception:
                    return 0

            scored_items.sort(key=lambda x: (-x[0], -get_modified_timestamp(x[1])))

            # **THRESHOLD-BASED FILTERING** (v21 improvement - replaces fixed top_k)
            items = [item for score, item in scored_items if score >= score_threshold]

            logger.info(
                f"Files meeting threshold ({score_threshold}): {len(items)} files selected"
            )

            if not items:
                logger.warning(
                    f"No items met threshold of {score_threshold}. Returning empty list to avoid polluting context."
                )
                # Return empty so we rely purely on Pinecone for this query
                items = []

        else:
            items = all_items[:5]

        return items

    except Exception as e:
        logger.error(
            f"Unexpected error in search: {type(e).__name__} - {str(e)}",
            exc_info=True,
        )
        raise


# --- 7. MAIN INTERFACE ---

# Import the spreadsheet engine (used by registry views)
from spreadsheet_engine import SpreadsheetEngine
import pandas as pd

# Function to load The National Registry from Google Sheets
@st.cache_resource
def load_national_registry():
    """Load The National Registry from Google Sheets"""
    try:
        # Use the existing Google Sheets credentials from session state
        if st.session_state.google_client is None:
            st.error(
                "❌ Google authentication not initialized. Please check your setup."
            )
            return None

        # Google Sheets URL
        sheet_url = "https://docs.google.com/spreadsheets/d/1YatiITZyi4ItFToYUIOHLQV_CBL-PJyt85HIc5DOF8U/edit?usp=drive_link"

        # Extract sheet ID from URL
        sheet_id = "1YatiITZyi4ItFToYUIOHLQV_CBL-PJyt85HIc5DOF8U"

        # Convert to CSV export URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

        # Load the data
        df = pd.read_csv(csv_url)

        st.session_state.spreadsheet_engine = SpreadsheetEngine(df)
        st.session_state.spreadsheet_name = "The National Registry"

        logger.info("✅ Loaded The National Registry from Google Drive")
        return st.session_state.spreadsheet_engine

    except Exception as e:
        logger.error(f"Error loading The National Registry: {str(e)}")
        st.error(f"❌ Error loading The National Registry: {str(e)}")
        return None

# Function to change view
def set_view(view_name):
    st.session_state.current_view = view_name
    st.rerun()

# Homepage branding (always show)
st.markdown("""
<div class="remcensus-brand">
    <div class="remcensus-title">🦁 'Remcensus</div>
    <div class="remcensus-tagline">Thinking is 'Rinking</div>
</div>
""", unsafe_allow_html=True)

# --- HOME VIEW ---
if st.session_state.current_view == "home":
    # Show button menu
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Ask the 'Remsearch", use_container_width=True):
            set_view("remsearch")
    with col2:
        if st.button("Search for 'Rinking Names", use_container_width=True):
            set_view("rinking_names")
    with col3:
        if st.button("Find a 'Layer", use_container_width=True):
            set_view("find_layer")
    with col4:
        if st.button("Generate an 'Uzzle", use_container_width=True):
            st.info("🎲 'Uzzle generator coming soon!")
    
    # Filter options section
    st.markdown("<div style='text-align: center; margin: 2rem 0; color: #666; font-size: 0.9rem;'>✨ Features: AI-Powered • Date Filtering • Smart Categorization • Content Extraction</div>", unsafe_allow_html=True)
    
    # Add some feature highlights
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📚 Protocol Archives")
        st.markdown("Search through comprehensive 'Remier League documentation")
    with col2:
        st.markdown("### 🔍 Smart Search")
        st.markdown("AI-powered search with context and date filtering")
    with col3:
        st.markdown("### 📊 Registry Access")
        st.markdown("Quick lookup of members and 'rinking names")

# --- ASK THE 'REMSEARCH VIEW ---
elif st.session_state.current_view == "remsearch":
    # Back button
    if st.button("← Back to Home"):
        set_view("home")
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Search bar with form to prevent triggering on every keystroke
    with st.form(key="remsearch_form", clear_on_submit=False):
        query = st.text_input("", placeholder="Ask the 'Remsearch", label_visibility="collapsed", key="remsearch_input")
        submit_button = st.form_submit_button("🔍 Search", use_container_width=False)

    if query and submit_button:
        with st.spinner("🌀 Triage in progress..."):
            search_process = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "category_detection": None,
                "date_constraints": None,
                "pinecone_results": [],
                "drive_results": [],
                "extracted_content": [],
                "llm_engine_used": None,
                "errors": [],
            }
    
            # Clear any stale session cache
            if "last_query" in st.session_state and st.session_state.last_query == query:
                logger.info("Query cache detected - forcing refresh...")
            st.session_state.last_query = query
    
            try:
                logger.info(f"Processing query: {query[: 100]}...")
    
                # Step 1: Pinecone retrieval
                logger.info("Step 1: Running Pinecone embedding and retrieval...")
                result = st.session_state.google_client.models.embed_content(
                    model="text-embedding-004", contents=query
                )
                search_results = st.session_state.pc_index.query(
                    vector=result.embeddings[0].values, top_k=5, include_metadata=True
                )
                context_text = ""
    
                for match in search_results["matches"]:
                    meta = match["metadata"]
                    context_text += f"Source: {meta.get('source', 'Unknown')}\nContent: {meta.get('text', '')}\n\n"
                    search_process["pinecone_results"].append(
                        {
                            "source": meta.get("source", "Unknown"),
                            "score": match.get("score", "N/A"),
                        }
                    )
    
                logger.info(f"Pinecone returned {len(search_results['matches'])} results")
    
                # Step 2: Google Drive search (background, with date filtering)
                logger.info("Step 2: Fetching from Google Drive (background)...")
                try:
                    if st.session_state.drive_service is None:
                        logger.warning("Drive service is None - skipping Drive search")
                        search_process["errors"].append("Drive service not initialized")
                    else:
                        logger.info("Extracting date constraints from query...")
                        date_constraints = extract_date_constraints_from_query(query)
                        search_process["date_constraints"] = {
                            k: str(v) if isinstance(v, datetime) else v
                            for k, v in date_constraints.items()
                        }
    
                        logger.info("Detecting category intent from query...")
                        category_match, subcategory_match, category_confidence = (
                            match_category(query.lower(), RACRL_FOLDER_MAP)
                        )
                        search_process["category_detection"] = {
                            "category": category_match,
                            "subcategory": subcategory_match,
                            "confidence": category_confidence,
                        }
    
                        logger.info(
                            "Searching Drive with threshold-based filtering and date constraints..."
                        )
                        drive_files = fetch_drive_recent_files(
                            "10B8EsEQ2TlzQP5ADD43TcDSs_xp3plj9",
                            search_query=query,
                            score_threshold=SCORE_THRESHOLD,
                        )
    
                        logger.info(f"Found {len(drive_files)} files")
    
                        if drive_files:
                            for idx, f in enumerate(drive_files):
                                search_process["drive_results"].append(
                                    {
                                        "rank": idx + 1,
                                        "name": f.get("name"),
                                        "mime_type": f.get("mimeType"),
                                        "modified": f.get("modifiedTimeISO"),
                                    }
                                )
    
                            # Extract content from TOP 3 files if enabled
                            if extract_content and len(drive_files) > 0:
                                logger.info("Extracting content from top files...")
                            
                                # Process up to 3 files
                                files_to_process = drive_files[:3] 
                            
                                for i, file_obj in enumerate(files_to_process):
                                    file_id = file_obj.get("id")
                                    file_name = file_obj.get("name")
                                    mime_type = file_obj.get("mimeType")
                                    # [NEW] Get the link for citations
                                    file_url = file_obj.get("webViewLink", "#") 

                                    content, success, error = extract_file_content(
                                        st.session_state.drive_service,
                                        file_id,
                                        mime_type,
                                        file_name,
                                    )

                                    if success:
                                        logger.info(f"Successfully extracted {file_name}")
                                        search_process["extracted_content"].append(
                                            {
                                                "filename": file_name,
                                                "mime_type": mime_type,
                                                "content_length": len(content),
                                            }
                                        )
                                        # Append to context with header AND Link for the AI to reference
                                        context_text += f"\n\n--- Source {i+1}: {file_name} (Link: {file_url}) ---\n{content[: 8000]}\n" 
                                    else:
                                        logger.error(f"Failed to extract {file_name}: {error}")
                                        search_process["errors"].append(
                                            f"Content extraction failed for {file_name}: {error}"
                                        )
    
                            # Add metadata (hidden from UI but in context)
                            context_text += "\n---\nGoogle Drive Files Found:\n"
                            for f in drive_files:
                                context_text += f"- {f.get('name')} (Modified: {f.get('modifiedTimeISO')})\n"
    
                        else:
                            logger.warning("No files found in Drive search")
                            search_process["errors"].append(
                                "No files found in Drive search"
                            )
    
                except Exception as e_drive:
                    logger.error(
                        f"Drive search failed: {type(e_drive).__name__} - {str(e_drive)}"
                    )
                    search_process["errors"].append(f"Drive search error: {str(e_drive)}")
    
                # Step 3: Generate response with CLEAN context
                logger.info("Step 3: Preparing context for LLM...")

                # We want to KEEP the Pinecone results (which are at the start) 
                # AND the extracted file content, but REMOVE the file list metadata at the end.
            
                metadata_start = context_text.find("\n---\nGoogle Drive Files Found:")
            
                if metadata_start != -1:
                    # Keep everything up to the metadata list
                    context_for_llm = context_text[:metadata_start]
                else:
                    # No metadata section found, use everything
                    context_for_llm = context_text
    
                # Remove any metadata markers
                context_for_llm = re.sub(
                    r"\n---\nGoogle Drive Files Found: .*",
                    "",
                    context_for_llm,
                    flags=re.DOTALL,
                )
    
                logger.info(
                    f"Context sent to LLM: {len(context_for_llm)} chars (primary file only)"
                )
    
                logger.info("Step 3: Generating response from LLM...")
                raw_text, engine_used, logs = generate_response(context_for_llm, query)
                search_process["llm_engine_used"] = engine_used
                final_answer = enforce_rem_lexicon(raw_text)
    
                # Display result in a card
                st.markdown(f"""
                <div class="result-card">
                    {final_answer}
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"🤖 Generated via: {engine_used}")
    
                logger.info(f"Query processing completed. Engine: {engine_used}")
    
                # Developer mode only
                if dev_mode:
                    with st.expander("🔍 DEVELOPER MODE - Search Analysis"):
                        st.markdown("### Search Process")
                        st.write(f"**Query:** {search_process['query']}")
                        st.write(f"**Timestamp:** {search_process['timestamp']}")
    
                        if search_process["date_constraints"]:
                            st.write("**Date Constraints:**")
                            for key, value in search_process["date_constraints"].items():
                                if value:
                                    st.write(f"  - {key}: {value}")
    
                        if search_process["category_detection"]:
                            st.write(
                                f"**Category Detected:** {search_process['category_detection']['category']}"
                            )
                            st.write(
                                f"**Subcategory:** {search_process['category_detection']['subcategory']}"
                            )
    
                        if search_process["drive_results"]:
                            st.markdown("#### Files Found")
                            for result in search_process["drive_results"]:
                                st.write(
                                    f"- **{result['name']}** ({result['mime_type']}, Modified:  {result.get('modified', 'N/A')})"
                                )
    
                        if search_process["extracted_content"]:
                            st.markdown("#### Extracted Content")
                            for content in search_process["extracted_content"]:
                                st.write(
                                    f"- **{content['filename']}** ({content['content_length']} chars)"
                                )
    
                        if search_process["errors"]:
                            st.markdown("#### Errors")
                            for error in search_process["errors"]:
                                st.error(error)
    
                        st.markdown("#### Export Data")
                        search_json = json.dumps(search_process, indent=2)
                        st.download_button(
                            label="📥 Download Search Process JSON",
                            data=search_json,
                            file_name=f"search_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )
    
            except Exception as e:
                logger.error(
                    f"Critical error:  {type(e).__name__} - {str(e)}", exc_info=True
                )
                search_process["errors"].append(f"Critical error: {str(e)}")
    
                st.error(f"⚠️ Error processing query: {str(e)}")
    
                if dev_mode:
                    with st.expander("🔴 Error Details"):
                        st.write(f"**Error Type:** {type(e).__name__}")
                        st.write(f"**Message:** {str(e)}")
                        st.code(traceback.format_exc())

# --- SEARCH FOR 'RINKING NAMES VIEW ---
elif st.session_state.current_view == "rinking_names":
    # Back button
    if st.button("← Back to Home"):
        set_view("home")
    
    st.markdown("---")
    st.markdown("### 📊 Registry Search")
    st.markdown("Search The National Registry for 'rinking names")
    
    # Load the registry automatically
    with st.spinner("📥 Loading The National Registry from Google Drive..."):
        engine = load_national_registry()

    if engine is not None:
        # Show basic info
        with st.expander("📋 Registry Information", expanded=False):
            report = engine.validate_data_quality()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Members", report["total_rows"])
            with col2:
                st.metric("Total Columns", report["total_columns"])
            with col3:
                st.metric("Duplicate Rows", report["duplicates"])

            if report["warnings"]:
                with st.expander("⚠️ Data Quality Warnings"):
                    for warning in report["warnings"][:5]:  # Show first 5 warnings
                        st.write(f"• {warning}")
            else:
                st.success("✅ No data quality issues detected")

        # Show only Tab 1: Search 'Rinking Names
        st.subheader("Search 'Rinking Names")
        st.write("Search for members by 'rinking name.")

        # Use form to prevent search on every keystroke
        with st.form(key="rinking_search_form", clear_on_submit=False):
            search_query = st.text_input(
                "Enter 'rinking name:",
                placeholder="e.g., 'Tree of Life'",
                key="rinking_search_query",
            )
            submit_button = st.form_submit_button("🔍 Search")

        if search_query and submit_button:
            with st.spinner("🔍 Searching... "):
                results = engine.search(search_query, "'rinking Name", min_score=0.50)

            if results:
                st.success(f"✅ Found {len(results)} match(es)")

                for i, result in enumerate(results):
                    with st.expander(
                        f"#{i+1} - {result.value} (Score: {result.score:.1%})",
                        expanded=(i == 0),
                    ):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Match Score", f"{result.score:.1%}")
                        with col2:
                            st.metric("Strategy", result.strategy.value.title())
                        with col3:
                            st.metric("Field", result.column)
                        with col4:
                            st.metric("Row ID", result.row_index)

                        st.divider()

                        row_data = engine.get_row_data(result.row_index)

                        rinking_name = engine.get_safe_value(
                            row_data, ["'rinking Name", "Drinking Name"]
                        )
                        full_name = engine.get_safe_value(row_data, ["Full Name"])
                        university = engine.get_safe_value(row_data, ["University"])
                        graduation_year = engine.get_safe_value(
                            row_data,
                            [
                                "Est. Year of Graduation",
                                "Est.   Year of Graduation",
                                "Est Year of Graduation",
                                "Estimated Year of Graduation",
                            ],
                        )

                        st.write("**Member Details:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"🎭 **'rinking Name:** {rinking_name}")
                            st.write(f"👤 **Full Name:** {full_name}")
                        with col2:
                            st.write(f"🎓 **University:** {university}")
                            st.write(f"📅 **Graduation:** {graduation_year}")

                        with st.expander("📋 Full Member Record"):
                            st.dataframe(
                                pd.DataFrame([row_data]).T, use_container_width=True
                            )
            else:
                st.warning("❌ No matches found.   Try a different search term.")
    else:
        st.error(
            "❌ Failed to load The National Registry.   Please check your internet connection."
        )

# --- FIND A 'LAYER VIEW ---
elif st.session_state.current_view == "find_layer":
    # Back button
    if st.button("← Back to Home"):
        set_view("home")
    
    st.markdown("---")
    st.markdown("### 📊 Registry Search")
    st.markdown("Search The National Registry for members")
    
    # Load the registry automatically
    with st.spinner("📥 Loading The National Registry from Google Drive..."):
        engine = load_national_registry()

    if engine is not None:
        # Show basic info
        with st.expander("📋 Registry Information", expanded=False):
            report = engine.validate_data_quality()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Members", report["total_rows"])
            with col2:
                st.metric("Total Columns", report["total_columns"])
            with col3:
                st.metric("Duplicate Rows", report["duplicates"])

            if report["warnings"]:
                with st.expander("⚠️ Data Quality Warnings"):
                    for warning in report["warnings"][:5]:  # Show first 5 warnings
                        st.write(f"• {warning}")
            else:
                st.success("✅ No data quality issues detected")

        # Create tabs for the 3 search options
        tab1, tab2, tab3 = st.tabs(
            [
                "👤 Name Lookup",
                "🎓 Search by University",
                "🔍 Search by Other Value",
            ]
        )

        # ========== TAB 1: NAME LOOKUP ==========
        with tab1:
            st.subheader("Search by Full Name")
            st.write("Find a member by their full name.")

            # Use form to prevent search on every keystroke
            with st.form(key="fullname_lookup_form", clear_on_submit=False):
                lookup_query = st.text_input(
                    "Enter full name:",
                    placeholder="e.g., 'Harry Foley'",
                    key="fullname_lookup_query",
                )
                submit_button = st.form_submit_button("🔍 Search")

            if lookup_query and submit_button:
                with st.spinner("🔍 Searching..."):
                    results = engine.search(lookup_query, "Full Name", min_score=0.50)

                if results:
                    st.success(f"✅ Found {len(results)} match(es)")

                    for i, result in enumerate(results):
                        with st.expander(
                            f"#{i+1} - {result.value} (Score: {result.score:.1%})",
                            expanded=(i == 0),
                        ):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Match Score", f"{result.score:.1%}")
                            with col2:
                                st.metric("Strategy", result.strategy.value.title())
                            with col3:
                                st.metric("Field", result.column)
                            with col4:
                                st.metric("Row ID", result.row_index)

                            st.divider()

                            row_data = engine.get_row_data(result.row_index)

                            rinking_name = engine.get_safe_value(
                                row_data, ["'rinking Name", "Drinking Name"]
                            )
                            full_name = engine.get_safe_value(row_data, ["Full Name"])
                            university = engine.get_safe_value(row_data, ["University"])
                            graduation_year = engine.get_safe_value(
                                row_data,
                                [
                                    "Est.  Year of Graduation",
                                    "Est.  Year of Graduation",
                                    "Est Year of Graduation",
                                    "Estimated Year of Graduation",
                                ],
                            )

                            st.write("**Member Details:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"🎭 **'rinking Name:** {rinking_name}")
                                st.write(f"👤 **Full Name:** {full_name}")
                            with col2:
                                st.write(f"🎓 **University:** {university}")
                                st.write(f"📅 **Graduation:** {graduation_year}")

                            with st.expander("📋 Full Member Record"):
                                st.dataframe(
                                    pd.DataFrame([row_data]).T, use_container_width=True
                                )
                else:
                    st.error("❌ No match found.  Try a different name.")

        # ========== TAB 2: SEARCH BY UNIVERSITY ==========
        with tab2:
            st.subheader("Search by University")
            st.write("Find all members from a specific university.")

            universities = sorted(
                [u for u in engine.df["University"].unique() if pd.notna(u)]
            )
            selected_uni = st.selectbox(
                "Select university:", universities, key="uni_filter_select"
            )

            if st.button("Go", key="uni_filter_go"):
                filtered_df = engine.df[engine.df["University"] == selected_uni]
                st.success(f"✅ Found {len(filtered_df)} member(s) from {selected_uni}")

                # Display results
                for idx, row in filtered_df.iterrows():
                    row_data = dict(row)
                    rinking_name = engine.get_safe_value(
                        row_data, ["'rinking Name", "Drinking Name"]
                    )
                    full_name = engine.get_safe_value(row_data, ["Full Name"])

                    with st.expander(f"{rinking_name} - {full_name}"):
                        university = engine.get_safe_value(row_data, ["University"])
                        graduation_year = engine.get_safe_value(
                            row_data,
                            [
                                "Est. Year of Graduation",
                                "Est.   Year of Graduation",
                                "Est Year of Graduation",
                                "Estimated Year of Graduation",
                            ],
                        )

                        st.write("**Member Details:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"🎭 **'rinking Name:** {rinking_name}")
                            st.write(f"👤 **Full Name:** {full_name}")
                        with col2:
                            st.write(f"🎓 **University:** {university}")
                            st.write(f"📅 **Graduation:** {graduation_year}")

                        with st.expander("📋 Full Record"):
                            st.dataframe(
                                pd.DataFrame([row_data]).T, use_container_width=True
                            )

        # ========== TAB 3: SEARCH BY OTHER VALUE ==========
        with tab3:
            st.subheader("Search by Other Value")
            st.write("Find members by various attributes.")

            # Columns to exclude
            exclude_cols = {"'rinking Name", "Drinking Name", "Full Name", "University"}
            available_cols = [col for col in engine.df.columns if col not in exclude_cols]

            selected_col = st.selectbox(
                "Select field:", available_cols, key="other_value_select"
            )

            if st.button("Go", key="other_value_go"):
                # Get unique values for the selected column
                unique_values = sorted(
                    [str(v) for v in engine.df[selected_col].unique() if pd.notna(v)]
                )

                if not unique_values:
                    st.warning(f"No values found in {selected_col}")
                else:
                    # Display all unique values
                    st.write(f"**Values in {selected_col}:**")

                    for value in unique_values:
                        # Filter rows that match this value
                        filtered_df = engine.df[
                            engine.df[selected_col].astype(str) == value
                        ]

                        # Special sorting for Current/'ast columns
                        if selected_col in [
                            "NHA",
                            "NHA Chair",
                            "RACRL",
                            "KHA",
                            "NSWHA",
                            "SAHA",
                            "VHA",
                            "WAHA",
                            "THA",
                        ]:
                            # Sort:  Current first, then 'ast by graduation year (most recent first)
                            def sort_key(row):
                                status = str(row.get(selected_col, ""))
                                graduation = row.get(
                                    "Est. Year of Graduation",
                                    row.get("Est.   Year of Graduation", 0),
                                )

                                # Try to convert to int for sorting
                                try:
                                    grad_year = (
                                        int(graduation)
                                        if graduation not in [None, "", "nan"]
                                        else 0
                                    )
                                except:
                                    grad_year = 0

                                # Current comes first (0), then 'ast sorted by year descending (1 + negative year)
                                if status == "Current":
                                    return (0, 0)
                                elif status == "'ast":
                                    return (1, -grad_year)
                                else:
                                    return (2, -grad_year)

                            filtered_df = filtered_df.copy()
                            filtered_df["sort_key"] = filtered_df.apply(sort_key, axis=1)
                            filtered_df = filtered_df.sort_values("sort_key").drop(
                                "sort_key", axis=1
                            )

                        else:
                            # Sort by graduation year (most recent first)
                            def get_graduation_year(row):
                                graduation = row.get(
                                    "Est. Year of Graduation",
                                    row.get("Est.  Year of Graduation", 0),
                                )
                                try:
                                    return (
                                        -int(graduation)
                                        if graduation not in [None, "", "nan"]
                                        else 0
                                    )
                                except:
                                    return 0

                            filtered_df = filtered_df.copy()
                            filtered_df["sort_key"] = filtered_df.apply(
                                get_graduation_year, axis=1
                            )
                            filtered_df = filtered_df.sort_values("sort_key").drop(
                                "sort_key", axis=1
                            )

                        # Display results grouped by value
                        with st.expander(f"**{value}** ({len(filtered_df)} member(s))"):
                            for idx, row in filtered_df.iterrows():
                                row_data = dict(row)

                                rinking_name = engine.get_safe_value(
                                    row_data, ["'rinking Name", "Drinking Name"]
                                )
                                full_name = engine.get_safe_value(row_data, ["Full Name"])
                                university = engine.get_safe_value(row_data, ["University"])
                                graduation_year = engine.get_safe_value(
                                    row_data,
                                    [
                                        "Est. Year of Graduation",
                                        "Est.   Year of Graduation",
                                        "Est Year of Graduation",
                                        "Estimated Year of Graduation",
                                    ],
                                )

                                with st.expander(f"{rinking_name} - {full_name}"):
                                    st.write("**Member Details:**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"🎭 **'rinking Name:** {rinking_name}")
                                        st.write(f"👤 **Full Name:** {full_name}")
                                    with col2:
                                        st.write(f"🎓 **University:** {university}")
                                        st.write(f"📅 **Graduation:** {graduation_year}")

                                    with st.expander("📋 Full Record"):
                                        st.dataframe(
                                            pd.DataFrame([row_data]).T,
                                            use_container_width=True,
                                        )

    else:
        st.error(
            "❌ Failed to load The National Registry.   Please check your internet connection."
        )

logger.info("Application render completed")
