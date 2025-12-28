import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
import re

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="'Remcensus",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SECURE CONNECTION
if 'init_done' not in st.session_state:
    try:
        pc = Pinecone(api_key=st.secrets["PINECONE_KEY"])
        st.session_state.pc_index = pc.Index(st.secrets["PINECONE_INDEX"])
        genai.configure(api_key=st.secrets["GEMINI_KEY"])
        st.session_state.init_done = True
    except Exception as e:
        st.error(f"🔐 Security Error: Keys missing. {e}")
        st.stop()

# --- 2. LINGUISTIC TRANSFORMATION ENGINE ---
def enforce_rem_lexicon(text):
    """Applies specific 'Remier League linguistic and numeric constraints."""
    text = re.sub(r'\bP(\w+)', r"'R\1", text)
    text = re.sub(r'\bp(\w+)', r"'r\1", text)
    replacements = {"Table": "'able", "table": "'able", "Drink": "'rink", "drink": "'rink"}
    for word, replacement in replacements.items():
        text = text.replace(word, replacement)
    num_map = [
        (r"\b444\b", "BJBJBJ"), (r"\bfour hundred and fourty four\b", "bon jovi hundred and bon jorty bon jovi"),
        (r"\b888\b", "TKTKTK"), (r"\beight hundred and eighty eight\b", "takahashi hundred and takahashity takahashi"),
        (r"\b400\b", "BJ00"), (r"\bfour hundred\b", "bon jovi hundred"),
        (r"\b800\b", "TK00"), (r"\beight hundred\b", "takahashi hundred"),
        (r"\b40\b", "BJ0"), (r"\bfourty\b", "bon jorty"),
        (r"\b80\b", "TK0"), (r"\beighty\b", "takahashity"),
        (r"\b14\b", "1BJ"), (r"\bfourteen\b", "bon jorteen"),
        (r"\b18\b", "1TK"), (r"\beighteen\b", "takahashiteen"),
        (r"\b4\b", "BJ"), (r"\bfour\b", "bon jovi"),
        (r"\b8\b", "TK"), (r"\beight\b", "takahashi"),
        (r"\b10\b", "IJ"), (r"\bten\b", "iku jo")
    ]
    for pattern, sub in num_map:
        text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    return text

# --- 3. DYNAMIC MODEL HANDSHAKE ---
def get_working_model():
    """Tries specific modern model IDs to bypass 404 versioning errors."""
    # Priority list for 2025 stability
    stable_ids = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-002",
        "gemini-2.0-flash",
        "gemini-pro"
    ]
    
    try:
        available = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for sid in stable_ids:
            if sid in available:
                return sid
        return available[0] if available else "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🦁 'Remcensus")
    st.caption("Archives of the 'ublic Library of the RACRL")
    st.markdown("---")
    st.success("✅ System Online")
    st.info("'rink and Learn")
    
    if 'model_id' not in st.session_state:
        st.session_state.model_id = get_working_model()
    st.caption(f"Protocol: {st.session_state.model_id}")

# --- 5. MAIN INTERFACE ---
st.markdown("## 🦁 'Remcensus")
query = st.text_input("Enter Query Parameters:", placeholder="e.g., Who is the