import streamlit as st
from google import genai as google_genai
from groq import Groq
from huggingface_hub import InferenceClient
from pinecone import Pinecone
import re
import json
from datetime import datetime

# Google Drive API imports (for metadata)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except Exception:
    # If google-api-python-client / google-auth not installed, we'll handle at runtime
    service_account = None
    build = None

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="'Remcensus",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar controls for Google Drive integration
st.sidebar.title("🦁 'Remcensus")
st.sidebar.success("✅ Protocol Discovery Active")
st.sidebar.markdown("---")
use_gdrive = st.sidebar.checkbox("Include Google Drive metadata in search", value=False)
drive_id_input = st.sidebar.text_input("Google Drive ID to use:", value="10B8EsEQ2TlzQP5ADD43TcDSs_xp3plj9")
gdrive_top_k = st.sidebar.number_input("Number of drive items to include (metadata only)", min_value=1, max_value=20, value=5, step=1)

# SECURE CONNECTION
if 'init_done' not in st.session_state:
    try:
        pc = Pinecone(api_key=st.secrets["PINECONE_KEY"])
        st.session_state.pc_index = pc.Index(st.secrets["PINECONE_INDEX"])
        st.session_state.google_client = google_genai.Client(api_key=st.secrets["GEMINI_KEY"])
        st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        st.session_state.hf_client = InferenceClient(api_key=st.secrets["HUGGINGFACE_KEY"])
        # Initialize Google Drive service if possible and configured
        st.session_state.drive_service = None
        if use_gdrive:
            if service_account is None or build is None:
                st.warning("Google Drive client libraries not available. Install `google-api-python-client` and `google-auth` to use Drive features.")
            else:
                if "GDRIVE_SERVICE_ACCOUNT_JSON" in st.secrets:
                    sa_json = st.secrets["GDRIVE_SERVICE_ACCOUNT_JSON"]
                    # Accept either a dict (from TOML nested secrets) or a JSON string
                    if isinstance(sa_json, dict):
                        creds_info = sa_json
                    else:
                        try:
                            creds_info = json.loads(sa_json)
                        except Exception as e:
                            raise ValueError("GDRIVE_SERVICE_ACCOUNT_JSON in secrets must be valid JSON or a secrets dict.") from e
                    scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
                    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
                    drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
                    st.session_state.drive_service = drive_service
                else:
                    st.warning("GDRIVE_SERVICE_ACCOUNT_JSON not found in Streamlit secrets. Drive metadata will be unavailable.")
        st.session_state.init_done = True
    except Exception as e:
        st.error(f"🔐 Security Error: {e}")
        st.stop()

# --- Drive metadata helper ---
def fetch_drive_recent_files(drive_id, top_k=5):
    """
    Fetch metadata for the most recent files in a shared drive (drive_id).
    Returns a list of dicts with selected metadata fields.
    """
    drive_service = st.session_state.get("drive_service")
    if drive_service is None:
        raise RuntimeError("Drive service not initialized. Ensure GDRIVE_SERVICE_ACCOUNT_JSON is set in secrets and the Drive libraries are installed.")
    try:
        # Use corpora='drive' and driveId to list files in a shared drive
        resp = drive_service.files().list(
            corpora="drive",
            driveId=drive_id,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            orderBy="createdTime desc",
            pageSize=top_k,
            fields="files(id,name,createdTime,modifiedTime,owners(mimeType,displayName,emailAddress),mimeType,webViewLink,size,description)"
        ).execute()
        files = resp.get("files", [])
        # Normalize datetime strings into ISO format for display
        for f in files:
            if "createdTime" in f:
                try:
                    dt = datetime.fromisoformat(f["createdTime"].replace("Z", "+00:00"))
                    f["createdTimeISO"] = dt.isoformat()
                except Exception:
                    f["createdTimeISO"] = f.get("createdTime")
        return files
    except Exception as e:
        raise

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
        text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    return text

# --- 3. UNIVERSAL SYSTEM PROMPT (REINFORCED) ---
SYSTEM_PROMPT = (
    "You are the Librarian of the 'Remier League. Conduct a silent background triage of the request.\n\n"
    "1. Do not mention Tiers or classification labels in your response.\n"
    "2. For simple identity queries, provide only an ontological definition. Do not teach mechanics unless specifically asked 'How to play'.\n"
    "3. DIDACTIC TEACHING: Only allowed for Basic Whiz, Antlers, Chow-Chow-Bang, Takahashi (1-3, 5-7), and Etiquette[cite: 20, 21, 37, 42, 45, 53].\n"
    "4. MANDATORY KILL-SWITCH: If the query mentions any of the following restricted terms (including shorthands), respond ONLY with the phrase: 'rink and learn.\n"
    "RESTRICTED TERMS: Beelze-bub-bub-bub, Bb, bzb, bzbz, Botsquali, Bsq, Bop, Kumquat, Kq, Kqs, Zoom, Kuon Kuon Chi Baa, KKXB, Viking Master, Bon Jovi, BJ, Takahashi, TK, Iku Jo, IJ, 4, 8, 9, 10[cite[...]\n"
    "5. Tone: Archival, bureaucratic."
)

# --- 4. TRIPLE-ENGINE HANDLER ---
def generate_response(context, query):
    debug_logs = []
    try:
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content, "Groq (Llama 3.3)", debug_logs
    except Exception as e:
        debug_logs.append(f"Groq: {str(e)}")
        try:
            response = st.session_state.google_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Context: {context}\n\nQuestion: {query}",
                config={'system_instruction': SYSTEM_PROMPT}
            )
            return response.text, "Gemini (1.5 Flash)", debug_logs
        except Exception as e_gem:
            debug_logs.append(f"Gemini: {str(e_gem)}")
            try:
                response = st.session_state.hf_client.chat_completion(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}],
                    max_tokens=800
                )
                return response.choices[0].message.content, "Hugging Face (Llama 3.2)", debug_logs
            except Exception as e_hf:
                debug_logs.append(f"HF: {str(e_hf)}")
                return "⚠️ SYSTEM FAILURE: All protocols failed.", "OFFLINE", debug_logs

# --- 5. MAIN INTERFACE ---
query = st.text_input("Enter Query Parameters:", placeholder="Search the archives...")

if query:
    with st.spinner("🌀 Triage in progress..."):
        try:
            # Shortcut: if user specifically asks for the most recent file in the drive, return a direct answer (faster)
            q_lower = query.lower()
            wants_most_recent = False
            if use_gdrive and ("most recent" in q_lower or "latest file" in q_lower or "newest file" in q_lower):
                wants_most_recent = True

            # Run embedding + pinecone retrieval as before
            result = st.session_state.google_client.models.embed_content(model="text-embedding-004", contents=query)
            search_results = st.session_state.pc_index.query(vector=result.embeddings[0].values, top_k=5, include_metadata=True)
            context_text = ""
            for match in search_results['matches']:
                meta = match['metadata']
                context_text += f"Source: {meta.get('source', 'Unknown')}\nContent: {meta.get('text', '')}\n\n"

            # If Drive metadata inclusion is enabled, fetch and append metadata to context_text
            if use_gdrive:
                try:
                    drive_files = fetch_drive_recent_files(drive_id_input, top_k=gdrive_top_k)
                    if drive_files:
                        # If the user directly asked for the most recent file, show it immediately
                        if wants_most_recent:
                            most_recent = drive_files[0]
                            # Present a concise, actionable answer
                            created = most_recent.get("createdTimeISO", most_recent.get("createdTime", "unknown"))
                            name = most_recent.get("name", "Unnamed")
                            link = most_recent.get("webViewLink", "No link available (insufficient perms)")
                            mime = most_recent.get("mimeType", "unknown")
                            st.success("Most recent file in the Drive (by created time):")
                            st.write(f"- Name: **{name}**")
                            st.write(f"- Created: {created}")
                            st.write(f"- Type: {mime}")
                            st.write(f"- Link: {link}")
                            # Also add drive metadata into context so LLMs can reference it if they continue
                            context_text += "\n---\nGoogleDrive Metadata (most recent items):\n"
                            for f in drive_files:
                                owners = ", ".join([o.get("displayName", o.get("emailAddress", "unknown")) for o in f.get("owners", [])]) if f.get("owners") else "unknown"
                                context_text += f"Name: {f.get('name')}\nCreated: {f.get('createdTimeISO')}\nOwners: {owners}\nLink: {f.get('webViewLink','')}\nMime: {f.get('mimeType')}\n\n"
                            # Continue to generate a fuller response below if desired
                        else:
                            # Append top-N drive metadata to the context for the model to use
                            context_text += "\n---\nGoogleDrive Metadata (top items by created time):\n"
                            for f in drive_files:
                                owners = ", ".join([o.get("displayName", o.get("emailAddress", "unknown")) for o in f.get("owners", [])]) if f.get("owners") else "unknown"
                                context_text += f"Name: {f.get('name')}\nCreated: {f.get('createdTimeISO')}\nOwners: {owners}\nLink: {f.get('webViewLink','')}\nMime: {f.get('mimeType')}\n\n"
                    else:
                        st.info("No files returned from the specified Drive (or insufficient permissions).")
                except Exception as e_drive:
                    st.warning(f"Google Drive metadata fetch failed: {e_drive}")

            raw_text, engine_used, logs = generate_response(context_text, query)
            final_answer = enforce_rem_lexicon(raw_text)
            st.caption(f"Generated via: {engine_used}")
            st.info(final_answer)
            if engine_used == "OFFLINE":
                with st.expander("🛠 Diagnostic Logs"):
                    for log in logs:
                        st.code(log)
        except Exception as e:
            st.error(f"⚠️ Critical Error: {e}")