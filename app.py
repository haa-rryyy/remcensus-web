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
    # Words starting with P (Replace P/p with apostrophe only)
    text = re.sub(r'\bP', "'", text)
    text = re.sub(r'\bp', "'", text)

    # Terminology Replacements
    replacements = {
        "Table": "'able", "table": "'able",
        "Drink": "'rink", "drink": "'rink"
    }
    for word, replacement in replacements.items():
        text = text.replace(word, replacement)

    # Numeric Substitutions
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
    """Prioritizes 1.5-flash series for stability and higher free-tier quotas."""
    # 2.5-flash is removed to avoid the 429 quota traps encountered.
    stable_ids = ["gemini-1.5-flash", "gemini-1.5-flash-002", "gemini-1.5-flash-001", "gemini-pro"]
    try:
        available = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for sid in stable_ids:
            if sid in available: return sid
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
query = st.text_input("Enter Query Parameters:", placeholder="e.g., Who is the most tone-deaf member of the NHA?")

if query:
    with st.spinner("🌀 Whizzing..."):
        try:
            result = genai.embed_content(model="models/text-embedding-004", content=query)
            search_results = st.session_state.pc_index.query(
                vector=result['embedding'], top_k=5, include_metadata=True
            )
            context_text = ""
            for match in search_results['matches']:
                meta = match['metadata']
                context_text += f"Source: {meta.get('source', 'Unknown')}\nContent: {meta.get('text', '')}\n\n"
            
            model = genai.GenerativeModel(st.session_state.model_id)
            
            prompt = f"""
            SYSTEM INSTRUCTION: You are the Librarian of the 'Remier League. 
            MANDATE: If the user provides a direct task (like writing a list), execute it precisely. 
            If the user asks a question, answer strictly based on the provided context.
            If context is required but missing, say "Data not found in the archives."
            Tone: Clinical, precise, bureaucratic.
            
            Context: {context_text}
            Question/Task: {query}
            """
            
            response = model.generate_content(prompt)
            final_answer = enforce_rem_lexicon(response.text)
        except Exception as e:
            if "429" in str(e):
                final_answer = "⚠️ System Error: Daily/Minute Quota exceeded. Please wait 60 seconds or try again later."
            else:
                final_answer = f"⚠️ System Error: {e}"
            search_results = {'matches': []}

    col1, col2 = st.columns([2, 1]) 
    with col1:
        st.subheader("📝 Consensus Summary")
        st.info(final_answer)
    with col2:
        st.subheader("📂 Reference Data")
        if search_results.get('matches'):
            for match in search_results['matches']:
                with st.expander(f"📄 {match['metadata'].get('source', 'Doc')}"):
                    st.write(match['metadata'].get('text', '')[:300] + "...")