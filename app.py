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

# Google Drive API imports
from google.auth.transport. requests import Request
from google.oauth2.service_account import Credentials
from google.api_core.exceptions import GoogleAPICallError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging. FileHandler('drive_service_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="'Remcensus",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        logger.debug("Attempting to read GDRIVE_SERVICE_ACCOUNT_JSON from Streamlit secrets...")
        
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
            required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in cred_content]
            
            if missing_fields:
                logger.error(f"Missing required fields in credentials: {missing_fields}")
                return None
            
            logger.debug(f"Credentials project_id: {cred_content.get('project_id')}")
            logger.debug(f"Credentials client_email: {cred_content.get('client_email')}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse credentials JSON: {str(e)}")
            return None
        except Exception as e:  
            logger.error(f"Unexpected error parsing credentials: {type(e).__name__} - {str(e)}")
            return None
        
        # Step 3: Create service account credentials
        logger.info("Creating service account credentials...")
        try:
            scopes = ['https://www.googleapis.com/auth/drive.metadata.readonly']
            logger.debug(f"Using scopes: {scopes}")
            
            credentials = Credentials. from_service_account_info(
                cred_content,
                scopes=scopes
            )
            logger.info("Service account credentials created successfully")
            logger.debug(f"Credentials type: {type(credentials)}")
            logger.debug(f"Credentials valid: {credentials.valid}")
            
        except ValueError as e:
            logger.error(f"Invalid service account credentials: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating credentials: {type(e).__name__} - {str(e)}")
            return None
        
        # Step 4: Refresh credentials if needed
        logger.info("Checking if credentials need refresh...")
        try:
            if not credentials.valid:
                logger. debug("Credentials not valid, attempting refresh...")
                credentials.refresh(Request())
                logger.info("Credentials refreshed successfully")
            else:
                logger.debug("Credentials are valid, no refresh needed")
        except Exception as e:
            logger.error(f"Failed to refresh credentials: {type(e).__name__} - {str(e)}")
            logger.debug("Continuing with unrefreshed credentials...")
        
        # Step 5: Build Drive service
        logger.info("Building Google Drive service...")
        try:
            service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
            logger.info("Google Drive service built successfully")
            logger.debug(f"Service type: {type(service)}")
            
        except HttpError as e:
            logger. error(f"HTTP error building Drive service: {e. resp. status} - {e.content}")
            return None
        except GoogleAPICallError as e:
            logger.error(f"Google API error building Drive service: {str(e)}")
            return None
        except Exception as e:  
            logger.error(f"Unexpected error building Drive service:  {type(e).__name__} - {str(e)}")
            return None
        
        # Step 6: Test the service
        logger.info("Testing Google Drive service with a basic API call...")
        try:
            results = service.files().list(
                spaces='drive',
                fields='files(id, name)',
                pageSize=1
            ).execute()
            
            logger.info("Google Drive service test successful")
            logger.debug(f"Test returned {len(results.get('files', []))} items")
            
        except HttpError as e:
            logger.error(f"HTTP error testing Drive service: {e.resp.status} - {e.content}")
            return None
        except Exception as e:
            logger.error(f"Error testing Drive service: {type(e).__name__} - {str(e)}")
            return None
        
        logger.info("=" * 80)
        logger.info("Google Drive Service Initialization COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return service
        
    except Exception as e:
        logger. critical(f"Unexpected error in initialize_drive_service: {type(e).__name__} - {str(e)}", exc_info=True)
        return None


# Sidebar controls for Google Drive integration
st.sidebar.title("🦁 'Remcensus")
st.sidebar.success("✅ Protocol Discovery Active")
st.sidebar.markdown("---")
use_gdrive = st.sidebar. checkbox("Include Google Drive metadata in search", value=False)
drive_id_input = st.sidebar.text_input("Google Drive ID to use:", value="10B8EsEQ2TlzQP5ADD43TcDSs_xp3plj9")
gdrive_top_k = st. sidebar.number_input("Number of drive items to include (metadata only)", min_value=1, max_value=20, value=5, step=1)

# SECURE CONNECTION
if 'init_done' not in st.session_state:
    try:
        logger.info("Initializing Pinecone and AI clients...")
        pc = Pinecone(api_key=st.secrets["PINECONE_KEY"])
        st.session_state.pc_index = pc.Index(st.secrets["PINECONE_INDEX"])
        st.session_state.google_client = google_genai. Client(api_key=st.secrets["GEMINI_KEY"])
        st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        st.session_state.hf_client = InferenceClient(api_key=st.secrets["HUGGINGFACE_KEY"])
        logger.info("Pinecone and AI clients initialized successfully")
        
        # Initialize Google Drive service
        logger.info("Initializing Google Drive service...")
        st.session_state.drive_service = initialize_drive_service()
        
        if st.session_state.drive_service:
            logger.info("Drive service initialized successfully")
        else:
            logger.warning("Drive service initialization returned None - Drive features will be unavailable")
        
        st.session_state.init_done = True
        logger.info("Session initialization completed")
        
    except Exception as e:
        logger.error(f"Security/Initialization Error: {e}", exc_info=True)
        st.error(f"🔐 Security Error: {e}")
        st.stop()

# --- 2. LINGUISTIC TRANSFORMATION ENGINE ---
def enforce_rem_lexicon(text):
    text = re.sub(r'\bP', "'", text)
    text = re.sub(r'\bp', "'", text)
    replacements = {"Table": "'able", "table": "'able", "Drink": "'rink", "drink": "'rink"}
    for word, replacement in replacements.items():
        text = text.replace(word, replacement)
    num_map = [
        (r"\b444\b", "BJBJBJ"), (r"\b888\b", "TKTKTK"),
        (r"\b400\b", "BJ00"), (r"\b800\b", "TK00"),
        (r"\b40\b", "BJ0"), (r"\b80\b", "TK0"),
        (r"\b14\b", "1BJ"), (r"\b18\b", "1TK"),
        (r"\b4\b", "BJ"), (r"\bfour\b", "bon jovi"),
        (r"\b8\b", "TK"), (r"\beight\b", "takahashi"),
        (r"\b10\b", "IJ"), (r"\bten\b", "iku jo")
    ]
    for pattern, sub in num_map:
        text = re.sub(pattern, sub, text, flags=re. IGNORECASE)
    return text

# --- 3. UNIVERSAL SYSTEM PROMPT (REINFORCED) ---
SYSTEM_PROMPT = (
    "You are a neutral information retrieval system for the 'Remier League archives.\n\n"
    "Provide only factual, direct answers based on the context provided.\n"
    "Do not add narrative descriptions, actions, or roleplay elements.\n"
    "Do not use stage directions like *pauses*, *scribbles*, etc.\n"
    "Keep responses concise and informational.\n\n"
    "1. Do not mention Tiers or classification labels in your response.\n"
    "2. For simple identity queries, provide only an ontological definition.   Do not teach mechanics unless specifically asked 'How to play'.\n"
    "3.  DIDACTIC TEACHING:   Only allowed for Basic Whiz, Antlers, Chow-Chow-Bang, Takahashi (1-3, 5-7), and Etiquette[cite: 20, 21, 37, 42, 45, 53].\n"
    "4. MANDATORY KILL-SWITCH: If the query mentions any of the following restricted terms (including shorthands), respond ONLY with the phrase:   'rink and learn.\n"
    "RESTRICTED TERMS: Beelze-bub-bub-bub, Bb, bzb, bzbz, Botsquali, Bsq, Bop, Kumquat, Kq, Kqs, Zoom, Kuon Kuon Chi Baa, KKXB, Viking Master, Bon Jovi, BJ, Takahashi, TK, Iku Jo, IJ, 4, 8, 9, 10\n"
    "5. Format:   Direct answers only. No narrative, actions, or roleplay."
)

# --- 4. TRIPLE-ENGINE HANDLER ---
def generate_response(context, query):
    debug_logs = []
    try:
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[{"role": "system", "content":  SYSTEM_PROMPT}, {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0]. message.content, "Groq (Llama 3.3)", debug_logs
    except Exception as e:
        debug_logs.append(f"Groq:  {str(e)}")
        try:
            response = st.session_state.google_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Context: {context}\n\nQuestion: {query}",
                config={'system_instruction':  SYSTEM_PROMPT}
            )
            return response.text, "Gemini (1.5 Flash)", debug_logs
        except Exception as e_gem:
            debug_logs.append(f"Gemini: {str(e_gem)}")
            try:
                response = st.session_state.hf_client.chat_completion(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=[{"role":  "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}],
                    max_tokens=800
                )
                return response. choices[0].message.content, "Hugging Face (Llama 3.2)", debug_logs
            except Exception as e_hf:
                debug_logs.append(f"HF: {str(e_hf)}")
                return "⚠️ SYSTEM FAILURE: All protocols failed.", "OFFLINE", debug_logs

# --- 5. GOOGLE DRIVE METADATA HELPER ---
def fetch_drive_recent_files(drive_id, top_k=5, search_query=None):
    """
    Fetch metadata for the most recent files in a drive or folder.
    Optionally filters by search query relevance.
    Returns a list of dicts with selected metadata fields.
    """
    drive_service = st.session_state.get("drive_service")
    if drive_service is None:
        logger.error("Drive service is None in fetch_drive_recent_files")
        raise RuntimeError("Drive service not initialized.  Ensure GDRIVE_SERVICE_ACCOUNT_JSON is set in secrets.")
    
    try:
        logger.info(f"Fetching recent files from drive/folder {drive_id}...")
        
        # Build query to exclude folders if search_query is provided
        query = f"'{drive_id}' in parents and trashed=false"
        if search_query:
            # Exclude folders when doing a specific search
            query += " and mimeType != 'application/vnd.google-apps.folder'"
            logger.debug(f"Filtering out folders for search query: {search_query}")
        
        # Query files within the folder/drive
        resp = drive_service.files().list(
            q=query,
            spaces='drive',
            orderBy="modifiedTime desc",
            pageSize=top_k * 3,  # Fetch more to filter by relevance
            fields="files(id,name,createdTime,modifiedTime,owners(displayName,emailAddress),mimeType,webViewLink,size,description)"
        ).execute()
        
        files = resp.get("files", [])
        logger.info(f"Successfully fetched {len(files)} files from folder/drive")
        
        # If search_query provided, score files by relevance
        if search_query:
            search_terms = search_query.lower().split()
            scored_files = []
            
            for f in files:
                # Score based on filename and description matches
                name_lower = f. get('name', '').lower()
                desc_lower = f.get('description', '').lower()
                score = 0
                
                for term in search_terms:
                    if term in name_lower:  
                        score += 3  # Higher weight for name matches
                    if term in desc_lower:
                        score += 1
                
                if score > 0:
                    scored_files. append((score, f))
            
            # Sort by score (descending) then by modified time
            scored_files.sort(key=lambda x: (-x[0], x[1].get('modifiedTime', '')), reverse=True)
            files = [f for _, f in scored_files[:top_k]]
            logger.info(f"Filtered to {len(files)} relevant files based on search query")
        
        # Normalize datetime strings into ISO format for display
        for f in files: 
            if "createdTime" in f:  
                try:
                    dt = datetime.fromisoformat(f["createdTime"].replace("Z", "+00:00"))
                    f["createdTimeISO"] = dt.isoformat()
                except Exception as dt_e:
                    logger. warning(f"Failed to parse datetime {f['createdTime']}: {dt_e}")
                    f["createdTimeISO"] = f. get("createdTime")
            
            if "modifiedTime" in f:
                try:
                    dt = datetime.fromisoformat(f["modifiedTime"]. replace("Z", "+00:00"))
                    f["modifiedTimeISO"] = dt.isoformat()
                except Exception as dt_e:
                    logger.warning(f"Failed to parse datetime {f['modifiedTime']}: {dt_e}")
                    f["modifiedTimeISO"] = f.get("modifiedTime")
        
        return files
        
    except HttpError as e:
        logger. error(f"HTTP error fetching drive files: {e.resp.status} - {e.content}")
        raise
    except GoogleAPICallError as e:
        logger.error(f"Google API error fetching drive files: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching drive files: {type(e).__name__} - {str(e)}", exc_info=True)
        raise

# --- 6. MAIN INTERFACE ---
query = st.text_input("Enter Query Parameters:", placeholder="Search the archives...")

if query: 
    with st.spinner("🌀 Triage in progress..."):
        try:
            logger.info(f"Processing query: {query[: 100]}...")
            
            # Shortcut:  if user specifically asks for the most recent file in the drive
            q_lower = query.lower()
            wants_most_recent = False
            if use_gdrive and ("most recent" in q_lower or "latest file" in q_lower or "newest file" in q_lower):
                wants_most_recent = True
                logger.debug("User query detected as requesting most recent file")

            # Run embedding + pinecone retrieval as before
            result = st.session_state.google_client.models.embed_content(model="text-embedding-004", contents=query)
            search_results = st.session_state.pc_index.query(vector=result. embeddings[0]. values, top_k=5, include_metadata=True)
            context_text = ""
            for match in search_results['matches']:
                meta = match['metadata']
                context_text += f"Source: {meta.get('source', 'Unknown')}\nContent: {meta.get('text', '')}\n\n"

            # If Drive metadata inclusion is enabled, fetch and append metadata to context_text
            if use_gdrive:  
                try:
                    logger.info("Fetching Google Drive metadata...")
                    # CHECK IF DRIVE_SERVICE IS NONE BEFORE CALLING
                    if st.session_state.drive_service is None:
                        logger.warning("Drive service is None - skipping Drive metadata")
                        st.warning("⚠️ Drive service not initialized. Drive metadata unavailable.")
                    else:
                        drive_files = fetch_drive_recent_files(drive_id_input, top_k=gdrive_top_k, search_query=query)
                        
                        if drive_files: 
                            logger.info(f"Retrieved {len(drive_files)} files from Drive")
                            
                            # If the user directly asked for the most recent file, show it immediately
                            if wants_most_recent:
                                most_recent = drive_files[0]
                                created = most_recent.get("createdTimeISO", most_recent.get("createdTime", "unknown"))
                                name = most_recent.get("name", "Unnamed")
                                link = most_recent.get("webViewLink", "No link available (insufficient perms)")
                                mime = most_recent.get("mimeType", "unknown")
                                
                                logger.info(f"Displaying most recent file: {name}")
                                st.success("Most recent file in the Drive (by created time):")
                                st.write(f"- Name: **{name}**")
                                st.write(f"- Created: {created}")
                                st.write(f"- Type: {mime}")
                                st.write(f"- Link: {link}")
                            
                            # Append drive metadata into context so LLMs can reference it
                            context_text += "\n---\nGoogleDrive Metadata (most recent items):\n"
                            for f in drive_files:
                                owners = ", ".join([o.get("displayName", o.get("emailAddress", "unknown")) for o in f.get("owners", [])]) if f.get("owners") else "unknown"
                                context_text += f"Name: {f. get('name')}\nCreated: {f.get('createdTimeISO')}\nOwners: {owners}\nLink: {f.get('webViewLink','')}\nMime: {f.get('mimeType')}\n\n"
                                logger.debug(f"Added to context: {f.get('name')}")
                        else:  
                            st.info("No files returned from the specified Drive (or insufficient permissions).")
                            logger.warning(f"No files returned from drive {drive_id_input}")
                        
                except Exception as e_drive:
                    logger.error(f"Google Drive metadata fetch failed: {type(e_drive).__name__} - {str(e_drive)}", exc_info=True)
                    st. error(f"❌ Google Drive metadata fetch failed:  {str(e_drive)}")

            logger.info("Generating response from LLM...")
            raw_text, engine_used, logs = generate_response(context_text, query)
            final_answer = enforce_rem_lexicon(raw_text)
            st.caption(f"Generated via: {engine_used}")
            st.info(final_answer)
            
            if engine_used == "OFFLINE":
                with st.expander("🛠 Diagnostic Logs"):
                    for log in logs:
                        st.code(log)
            
            logger.info(f"Query processing completed. Engine:  {engine_used}")
            
        except Exception as e:
            logger.error(f"Critical error processing query: {type(e).__name__} - {str(e)}", exc_info=True)
            st.error(f"⚠️ Critical Error: {e}")

logger.info("Application render completed")