import streamlit as st
from google import genai as google_genai
from groq import Groq
from huggingface_hub import InferenceClient
from pinecone import Pinecone
import re
import json
from datetime import datetime
import logging
import sys
import os
import traceback

# Google Drive API imports
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.api_core.exceptions import GoogleAPICallError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging with detailed formatting
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

# --- FOLDER STRUCTURE MAP (Update this when Drive changes) ---
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


# --- GOOGLE DRIVE SERVICE INITIALIZATION ---
def initialize_drive_service():
    """
    Initialize Google Drive service with detailed error logging and debugging.

    Returns:
        googleapiclient.discovery.Resource: The Drive service resource, or None if initialization fails.
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting Google Drive Service Initialization")
        logger.info("=" * 80)

        # Step 1: Check Streamlit secrets for credentials
        logger.debug(
            "Attempting to read GDRIVE_SERVICE_ACCOUNT_JSON from Streamlit secrets..."
        )

        try:
            cred_json = st.secrets.get("GDRIVE_SERVICE_ACCOUNT_JSON")
        except Exception as e:
            logger.error(f"Failed to access Streamlit secrets: {str(e)}")
            cred_json = None

        if not cred_json:
            logger.error("GDRIVE_SERVICE_ACCOUNT_JSON not found in Streamlit secrets")
            return None

        logger.info("GDRIVE_SERVICE_ACCOUNT_JSON found in Streamlit secrets")

        # Step 2: Parse credentials (handle both string and dict)
        logger.debug("Parsing credentials JSON...")
        try:
            if isinstance(cred_json, dict):
                cred_content = cred_json
                logger.debug("Credentials loaded as dict from secrets")
            elif isinstance(cred_json, str):
                cred_content = json.loads(cred_json)
                logger.debug("Credentials parsed from JSON string")
            else:
                logger.error(f"Unexpected credentials type: {type(cred_json)}")
                return None

            logger.debug(f"Credentials file structure: {list(cred_content.keys())}")

            # Validate required fields
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
                logger.error(
                    f"Missing required fields in credentials: {missing_fields}"
                )
                return None

            logger.debug(f"Credentials project_id: {cred_content.get('project_id')}")
            logger.debug(
                f"Credentials client_email: {cred_content.get('client_email')}"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse credentials JSON: {str(e)}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error parsing credentials: {type(e).__name__} - {str(e)}"
            )
            return None

        # Step 3: Create service account credentials
        logger.info("Creating service account credentials...")
        try:
            scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
            logger.debug(f"Using scopes: {scopes}")

            credentials = Credentials.from_service_account_info(
                cred_content, scopes=scopes
            )
            logger.info("Service account credentials created successfully")
            logger.debug(f"Credentials type: {type(credentials)}")
            logger.debug(f"Credentials valid: {credentials.valid}")

        except ValueError as e:
            logger.error(f"Invalid service account credentials: {str(e)}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error creating credentials: {type(e).__name__} - {str(e)}"
            )
            return None

        # Step 4: Refresh credentials if needed
        logger.info("Checking if credentials need refresh...")
        try:
            if not credentials.valid:
                logger.debug("Credentials not valid, attempting refresh...")
                credentials.refresh(Request())
                logger.info("Credentials refreshed successfully")
            else:
                logger.debug("Credentials are valid, no refresh needed")
        except Exception as e:
            logger.error(
                f"Failed to refresh credentials: {type(e).__name__} - {str(e)}"
            )
            logger.debug("Continuing with unrefreshed credentials...")

        # Step 5: Build Drive service
        logger.info("Building Google Drive service...")
        try:
            service = build(
                "drive", "v3", credentials=credentials, cache_discovery=False
            )
            logger.info("Google Drive service built successfully")
            logger.debug(f"Service type: {type(service)}")

        except HttpError as e:
            logger.error(
                f"HTTP error building Drive service: {e.resp.status} - {e.content}"
            )
            return None
        except GoogleAPICallError as e:
            logger.error(f"Google API error building Drive service: {str(e)}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error building Drive service:  {type(e).__name__} - {str(e)}"
            )
            return None

        # Step 6: Test the service
        logger.info("Testing Google Drive service with a basic API call...")
        try:
            results = (
                service.files()
                .list(spaces="drive", fields="files(id, name)", pageSize=1)
                .execute()
            )

            logger.info("Google Drive service test successful")
            logger.debug(f"Test returned {len(results.get('files', []))} items")

        except HttpError as e:
            logger.error(
                f"HTTP error testing Drive service: {e.resp.status} - {e.content}"
            )
            return None
        except Exception as e:
            logger.error(f"Error testing Drive service: {type(e).__name__} - {str(e)}")
            return None

        logger.info("=" * 80)
        logger.info("Google Drive Service Initialization COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return service

    except Exception as e:
        logger.critical(
            f"Unexpected error in initialize_drive_service: {type(e).__name__} - {str(e)}",
            exc_info=True,
        )
        return None


# Sidebar controls for Google Drive integration
st.sidebar.title("🦁 'Remcensus")
st.sidebar.success("✅ Protocol Discovery Active")
st.sidebar.markdown("---")
use_gdrive = st.sidebar.checkbox("Include Google Drive metadata in search", value=False)
drive_id_input = st.sidebar.text_input(
    "Google Drive ID to use:", value="10B8EsEQ2TlzQP5ADD43TcDSs_xp3plj9"
)
gdrive_top_k = st.sidebar.number_input(
    "Number of drive items to include (metadata only)",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Developer Mode")
dev_mode = st.sidebar.checkbox("Enable Developer Mode", value=True)

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

        # Initialize Google Drive service
        logger.info("Initializing Google Drive service...")
        st.session_state.drive_service = initialize_drive_service()

        if st.session_state.drive_service:
            logger.info("Drive service initialized successfully")
        else:
            logger.warning(
                "Drive service initialization returned None - Drive features will be unavailable"
            )

        st.session_state.init_done = True
        logger.info("Session initialization completed")

    except Exception as e:
        logger.error(f"Security/Initialization Error: {e}", exc_info=True)
        st.error(f"🔐 Security Error: {e}")
        st.stop()


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


# --- 3.UNIVERSAL SYSTEM PROMPT (REINFORCED) ---
SYSTEM_PROMPT = (
    "You are a neutral information retrieval system for the 'Remier League archives.\n\n"
    "Provide only factual, direct answers based on the context provided.\n"
    "Do not add narrative descriptions, actions, or roleplay elements.\n"
    "Do not use stage directions like *pauses*, *scribbles*, etc.\n"
    "Keep responses concise and informational.\n\n"
    "1.Do not mention Tiers or classification labels in your response.\n"
    "2.For simple identity queries, provide only an ontological definition.Do not teach mechanics unless specifically asked 'How to play'.\n"
    "3.DIDACTIC TEACHING:  Only allowed for Basic Whiz, Antlers, Chow-Chow-Bang, Takahashi (1-3, 5-7), and Etiquette[cite: 20, 21, 37, 42, 45, 53].\n"
    "4.MANDATORY KILL-SWITCH: If the query mentions any of the following restricted terms (including shorthands), respond ONLY with the phrase: 'rink and learn.\n"
    "RESTRICTED TERMS:  Beelze-bub-bub-bub, Bb, bzb, bzbz, Botsquali, Bsq, Bop, Kumquat, Kq, Kqs, Zoom, Kuon Kuon Chi Baa, KKXB, Viking Master, Bon Jovi, BJ, Takahashi, TK, Iku Jo, IJ, 4, 8, 9, 10\n"
    "5.Format:  Direct answers only.No narrative, actions, or roleplay."
)


# --- 4.TRIPLE-ENGINE HANDLER ---
def generate_response(context, query):
    debug_logs = []
    try:
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ],
            model="llama-3.3-70b-versatile",
        )
        return (
            chat_completion.choices[0].message.content,
            "Groq (Llama 3.3)",
            debug_logs,
        )
    except Exception as e:
        debug_logs.append(f"Groq:   {str(e)}")
        try:
            response = st.session_state.google_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Context: {context}\n\nQuestion: {query}",
                config={"system_instruction": SYSTEM_PROMPT},
            )
            return response.text, "Gemini (1.5 Flash)", debug_logs
        except Exception as e_gem:
            debug_logs.append(f"Gemini: {str(e_gem)}")
            try:
                response = st.session_state.hf_client.chat_completion(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Context: {context}\n\nQuestion: {query}",
                        },
                    ],
                    max_tokens=800,
                )
                return (
                    response.choices[0].message.content,
                    "Hugging Face (Llama 3.2)",
                    debug_logs,
                )
            except Exception as e_hf:
                debug_logs.append(f"HF: {str(e_hf)}")
                return (
                    "⚠️ SYSTEM FAILURE:   All protocols failed.",
                    "OFFLINE",
                    debug_logs,
                )


# --- 5.FOLDER STRUCTURE INTELLIGENCE ---
def match_category(query_lower, folder_map):
    """
    Intelligently match query to folder categories.
    Returns tuple:  (category_name, subcategory_name or None, confidence_score)
    """
    best_match = (None, None, 0)

    # Check subcategories FIRST (more specific)
    for category, info in folder_map.items():
        if "subcategories" in info:
            for subcat_name, subcat_info in info["subcategories"].items():
                subcat_keywords = subcat_info.get("keywords", [])
                for keyword in subcat_keywords:
                    if keyword in query_lower:
                        confidence = len(keyword) / len(query_lower)
                        if confidence > best_match[2]:
                            best_match = (category, subcat_name, confidence)

    # Check main categories ONLY if no subcategory matched
    if best_match[2] == 0:
        for category, info in folder_map.items():
            category_keywords = info.get("keywords", [])
            for keyword in category_keywords:
                if keyword in query_lower:
                    confidence = len(keyword) / len(query_lower)
                    if confidence > best_match[2]:
                        best_match = (category, None, confidence)

    return best_match


def extract_year(folder_name):
    """Extract year from folder name and return priority."""
    folder_lower = folder_name.lower()

    for year_pattern, priority in YEAR_PRIORITY.items():
        if year_pattern.lower() in folder_lower:
            return priority

    return 999


# --- 6. GOOGLE DRIVE METADATA HELPER ---
def fetch_drive_recent_files(drive_id, top_k=5, search_query=None):
    """
    Fetch metadata with intelligent folder structure awareness.
    Uses RACRL_FOLDER_MAP to understand hierarchy and prioritize results.
    Returns ONLY files, not folders (unless explicitly requested).
    """
    drive_service = st.session_state.get("drive_service")
    if drive_service is None:
        logger.error("Drive service is None in fetch_drive_recent_files")
        raise RuntimeError(
            "Drive service not initialized.  Ensure GDRIVE_SERVICE_ACCOUNT_JSON is set in secrets."
        )

    try:
        logger.info(
            f"Fetching recent files from drive/folder {drive_id} with intelligent folder structure..."
        )

        all_items = []
        folders_to_search = [drive_id]
        searched_folders = set()
        folder_count = 0

        logger.info("Starting recursive folder traversal...")
        while folders_to_search:
            current_folder = folders_to_search.pop(0)

            if current_folder in searched_folders:
                continue

            searched_folders.add(current_folder)
            folder_count += 1
            logger.info(f"[FOLDER #{folder_count}] Searching:  {current_folder}")

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
                logger.info(f"  -> Found {len(items)} items")

                for item in items:
                    name = item.get("name", "UNKNOWN")
                    mime = item.get("mimeType", "UNKNOWN")
                    is_folder = mime == "application/vnd.google-apps.folder"

                    if is_folder:
                        logger.info(f"    [FOLDER] {name}")
                        folders_to_search.append(item["id"])
                    else:
                        logger.info(f"    [FILE] {name}")
                        all_items.append(item)

            except HttpError as e:
                logger.warning(f"Error searching folder {current_folder}: {e}")
                continue

        logger.info(
            f"Search complete:  Found {len(all_items)} files across {len(searched_folders)} folders"
        )

        # Display debug info in Streamlit UI
        st.write(f"📊 **Folders searched:** {len(searched_folders)}")
        st.write(f"📁 **Total files found:** {len(all_items)}")
        st.write("**File names found:**")
        for file_item in all_items:
            st.write(f"  - {file_item.get('name')}")

        if len(all_items) == 0:
            logger.warning("No files found in the entire folder hierarchy!")

        all_items.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)

        if search_query: 
            search_terms = [
                term for term in search_query.lower().split() if len(term) > 2
    ]
    scored_items = []

    if search_query:
        search_terms = [
            term for term in search_query.lower().split() if len(term) > 2
        ]
        scored_items = []

        logger.info(
            f"Scoring {len(all_items)} files against search terms: {search_terms}"
        )

        for item in all_items:
            name_lower = item.get("name", "").lower()
            desc_lower = (
                item.get("description", "").lower()
                if item.get("description")
                else ""
            )
            score = 0

            for term in search_terms:
                if term in name_lower: 
                    score += 3
                if term in desc_lower:
                    score += 1

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

            scored_items.append((score, item))
            if score > 0:
                logger.debug(f"File: {item.get('name')} - Score: {score}")

        scored_items.sort(
            key=lambda x: (-x[0], x[1].get("modifiedTime", "")), reverse=True
        )
        items = [item for _, item in scored_items[:top_k]]
        logger.info(f"Top {len(items)} files after intelligent scoring")
    else:
        items = all_items[:top_k]
        logger.debug(
            f"Returning top {len(items)} most recent items (no search query)"
        )

    for item in items:
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

    return items

    except HttpError as e:
        logger.error(
            f"HTTP error fetching drive items: {e.resp.status} - {e.content}"
        )
        raise
    except GoogleAPICallError as e:
        logger.error(f"Google API error fetching drive items: {str(e)}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error fetching drive items: {type(e).__name__} - {str(e)}",
            exc_info=True,
        )
        raise


# --- 7.MAIN INTERFACE ---
query = st.text_input("Enter Query Parameters:", placeholder="Search the archives...")

if query:
    with st.spinner("🌀 Triage in progress..."):
        # Initialize search process tracking
        search_process = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "use_gdrive": use_gdrive,
            "drive_id": drive_id_input,
            "gdrive_top_k": gdrive_top_k,
            "category_detection": None,
            "pinecone_results": [],
            "drive_results": [],
            "context_sources": [],
            "llm_engine_used": None,
            "errors": [],
        }

        try:
            logger.info(f"Processing query: {query[: 100]}...")

            # Shortcut:   if user specifically asks for the most recent file in the drive
            q_lower = query.lower()
            wants_most_recent = False
            if use_gdrive and (
                "most recent" in q_lower
                or "latest file" in q_lower
                or "newest file" in q_lower
            ):
                wants_most_recent = True
                logger.debug("User query detected as requesting most recent file")
                search_process["wants_most_recent"] = True

            # Run embedding + pinecone retrieval
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
                        "content_preview": meta.get("text", "")[:100] + "...",
                    }
                )

            logger.info(f"Pinecone returned {len(search_results['matches'])} results")

            # If Drive metadata inclusion is enabled, fetch and append metadata to context_text
            if use_gdrive:
                try:
                    logger.info("Step 2: Fetching Google Drive metadata...")
                    # CHECK IF DRIVE_SERVICE IS NONE BEFORE CALLING
                    if st.session_state.drive_service is None:
                        logger.warning(
                            "Drive service is None - skipping Drive metadata"
                        )
                        st.warning(
                            "⚠️ Drive service not initialized.Drive metadata unavailable."
                        )
                        search_process["errors"].append("Drive service not initialized")
                    else:
                        logger.info("Step 2a: Detecting category intent from query...")
                        category_match, subcategory_match, category_confidence = (
                            match_category(query.lower(), RACRL_FOLDER_MAP)
                        )
                        search_process["category_detection"] = {
                            "category": category_match,
                            "subcategory": subcategory_match,
                            "confidence": category_confidence,
                        }
                        logger.info(
                            f"Category detected: {category_match} > {subcategory_match} (confidence: {category_confidence:.2f})"
                        )

                        logger.info("Step 2b: Recursively searching Drive folders...")
                        drive_files = fetch_drive_recent_files(
                            drive_id_input, top_k=gdrive_top_k, search_query=query
                        )

                        # ADD THIS DEBUG OUTPUT
                        st.write("🔍 DEBUG INFO:")
                        st.write(
                            f"Number of files returned: {len(drive_files) if drive_files else 0}"
                        )
                        if drive_files:
                            st.write("Files found:")
                            for f in drive_files:
                                st.write(f"  - {f.get('name')} ({f.get('mimeType')})")

                        if drive_files:
                            logger.info(
                                f"Step 2c: Retrieved {len(drive_files)} files from Drive"
                            )

                            for idx, f in enumerate(drive_files):
                                search_process["drive_results"].append(
                                    {
                                        "rank": idx + 1,
                                        "name": f.get("name"),
                                        "mime_type": f.get("mimeType"),
                                        "created": f.get("createdTimeISO"),
                                        "modified": f.get("modifiedTimeISO"),
                                        "link": f.get("webViewLink"),
                                    }
                                )

                            # If the user directly asked for the most recent file, show it immediately
                            if wants_most_recent:
                                most_recent = drive_files[0]
                                created = most_recent.get(
                                    "createdTimeISO",
                                    most_recent.get("createdTime", "unknown"),
                                )
                                name = most_recent.get("name", "Unnamed")
                                link = most_recent.get(
                                    "webViewLink",
                                    "No link available (insufficient perms)",
                                )
                                mime = most_recent.get("mimeType", "unknown")

                                logger.info(f"Displaying most recent file: {name}")
                                st.success(
                                    "Most recent file in the Drive (by relevance & recency):"
                                )
                                st.write(f"- Name: **{name}**")
                                st.write(f"- Created: {created}")
                                st.write(f"- Type: {mime}")
                                st.write(f"- Link: {link}")

                            # Append drive metadata into context so LLMs can reference it
                            context_text += (
                                "\n---\nGoogleDrive Metadata (most relevant items):\n"
                            )
                            for f in drive_files:
                                owners = (
                                    ", ".join(
                                        [
                                            o.get(
                                                "displayName",
                                                o.get("emailAddress", "unknown"),
                                            )
                                            for o in f.get("owners", [])
                                        ]
                                    )
                                    if f.get("owners")
                                    else "unknown"
                                )
                                context_text += f"Name: {f.get('name')}\nCreated: {f.get('createdTimeISO')}\nOwners: {owners}\nLink: {f.get('webViewLink', '')}\nMime: {f.get('mimeType')}\n\n"
                                logger.debug(f"Added to context: {f.get('name')}")
                        else:
                            st.info(
                                "No files returned from the specified Drive (or insufficient permissions)."
                            )
                            logger.warning(
                                f"No files returned from drive {drive_id_input}"
                            )
                            search_process["errors"].append(
                                "No files returned from Drive"
                            )

                except Exception as e_drive:
                    logger.error(
                        f"Google Drive metadata fetch failed: {type(e_drive).__name__} - {str(e_drive)}",
                        exc_info=True,
                    )
                    search_process["errors"].append(
                        f"Drive fetch error: {type(e_drive).__name__} - {str(e_drive)}"
                    )

                    error_msg = f"❌ Google Drive metadata fetch failed: {str(e_drive)}"
                    st.error(error_msg)

                    # Developer mode error report
                    if dev_mode:
                        with st.expander("🔴 DEVELOPER ERROR REPORT - DRIVE FETCH"):
                            st.markdown("### Error Details")
                            st.write(f"**Error Type:** `{type(e_drive).__name__}`")
                            st.write(f"**Error Message:** {str(e_drive)}")
                            st.write(f"**Drive ID:** `{drive_id_input}`")
                            st.write(f"**Query:** `{query}`")
                            st.write(f"**Search Query Enabled:** {use_gdrive}")
                            st.write(f"**Top-K:** {gdrive_top_k}")

                            # Show stack trace
                            st.markdown("### Stack Trace")
                            st.code(traceback.format_exc())

            logger.info("Step 3: Generating response from LLM...")
            raw_text, engine_used, logs = generate_response(context_text, query)
            search_process["llm_engine_used"] = engine_used
            final_answer = enforce_rem_lexicon(raw_text)
            st.caption(f"Generated via: {engine_used}")
            st.info(final_answer)

            if engine_used == "OFFLINE":
                with st.expander("🛠 Diagnostic Logs"):
                    for log in logs:
                        st.code(log)

            logger.info(f"Step 4: Query processing completed.Engine:   {engine_used}")

            # Always show developer mode search process details
            if dev_mode:
                with st.expander("🔍 DEVELOPER MODE - SEARCH PROCESS ANALYSIS"):
                    st.markdown("### Search Process Timeline")

                    # Query information
                    st.markdown("#### 1️⃣ Query Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Query:** `{search_process['query']}`")
                        st.write(f"**Timestamp:** {search_process['timestamp']}")
                    with col2:
                        st.write(
                            f"**Drive Search Enabled:** {search_process['use_gdrive']}"
                        )
                        st.write(f"**Drive ID:** `{search_process['drive_id']}`")

                    # Category detection
                    if search_process["category_detection"]:
                        st.markdown("#### 2️⃣ Category Detection")
                        cat_det = search_process["category_detection"]
                        st.write(f"**Primary Category:** {cat_det['category']}")
                        st.write(f"**Subcategory:** {cat_det['subcategory'] or 'None'}")
                        st.write(
                            f"**Detection Confidence:** {cat_det['confidence']:.2%}"
                        )

                    # Pinecone results
                    if search_process["pinecone_results"]:
                        st.markdown("#### 3️⃣ Pinecone Vector Search Results")
                        for idx, result in enumerate(
                            search_process["pinecone_results"], 1
                        ):
                            with st.container():
                                st.write(f"**Result {idx}**")
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"Source: `{result['source']}`")
                                    st.write(f"Preview: {result['content_preview']}")
                                with col2:
                                    st.write(f"Score: {result['score']}")

                    # Drive results
                    if search_process["drive_results"]:
                        st.markdown(
                            "#### 4️⃣ Google Drive Search Results (Ranked by Algorithm)"
                        )
                        for result in search_process["drive_results"]:
                            with st.container():
                                st.write(f"**#{result['rank']}** - {result['name']}")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"Type: `{result['mime_type']}`")
                                    st.write(f"Created: {result['created']}")
                                with col2:
                                    st.write(f"Modified: {result['modified']}")
                                st.write(f"Link: {result['link']}")

                    # LLM Engine
                    st.markdown("#### 5️⃣ LLM Processing")
                    st.write(f"**Engine Used:** {search_process['llm_engine_used']}")
                    st.write(
                        f"**Context Sources Combined:** {len(search_process['pinecone_results']) + len(search_process['drive_results'])} sources"
                    )

                    # Errors (if any)
                    if search_process["errors"]:
                        st.markdown("#### ⚠️ Errors Encountered")
                        for error in search_process["errors"]:
                            st.error(error)
                    else:
                        st.success("✅ No errors encountered during search process")

                    # Export button for debugging
                    st.markdown("#### 📊 Export Search Data")
                    search_json = json.dumps(search_process, indent=2)
                    st.download_button(
                        label="📥 Download Search Process JSON",
                        data=search_json,
                        file_name=f"search_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

        except Exception as e:
            logger.error(
                f"Critical error processing query: {type(e).__name__} - {str(e)}",
                exc_info=True,
            )
            search_process["errors"].append(
                f"Critical error: {type(e).__name__} - {str(e)}"
            )

            error_msg = f"⚠️ Critical Error: {e}"
            st.error(error_msg)

            # Developer mode error report
            if dev_mode:
                with st.expander("🔴 DEVELOPER ERROR REPORT - CRITICAL"):
                    st.markdown("### Critical Error Details")
                    st.write(f"**Error Type:** `{type(e).__name__}`")
                    st.write(f"**Error Message:** {str(e)}")
                    st.write(f"**Query:** `{query}`")
                    st.write(f"**Drive Metadata Enabled:** {use_gdrive}")
                    st.write(f"**Drive ID:** `{drive_id_input}`")

                    # Show all relevant session state
                    st.markdown("### Session State")
                    st.write(
                        f"**Drive Service Initialized:** {st.session_state.drive_service is not None}"
                    )
                    st.write(
                        f"**Pinecone Index Ready:** {hasattr(st.session_state, 'pc_index')}"
                    )
                    st.write(
                        f"**Google Client Ready:** {hasattr(st.session_state, 'google_client')}"
                    )
                    st.write(
                        f"**Groq Client Ready:** {hasattr(st.session_state, 'groq_client')}"
                    )
                    st.write(
                        f"**HuggingFace Client Ready:** {hasattr(st.session_state, 'hf_client')}"
                    )

                    # Show stack trace
                    st.markdown("### Stack Trace")
                    st.code(traceback.format_exc())

                    # Show search process so far
                    st.markdown("### Search Process Before Error")
                    st.json(search_process)

logger.info("Application render completed")
