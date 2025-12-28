import streamlit as st
from google import genai as google_genai
from groq import Groq
from huggingface_hub import InferenceClient
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
        st.session_state.google_client = google_genai.Client(api_key=st.secrets["GEMINI_KEY"])
        st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        st.session_state.hf_client = InferenceClient(api_key=st.secrets["HUGGINGFACE_KEY"])
        st.session_state.init_done = True
    except Exception as e:
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
        text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    return text

# --- 3. UNIVERSAL SYSTEM PROMPT (REINFORCED) ---
SYSTEM_PROMPT = (
    "You are the Librarian of the 'Remier League. Conduct a silent background triage of the request.\n\n"
    "1. Do not mention Tiers or classification labels in your response.\n"
    "2. For simple identity queries, provide only an ontological definition. Do not teach mechanics unless specifically asked 'How to play'.\n"
    "3. DIDACTIC TEACHING: Only allowed for Basic Whiz, Antlers, Chow-Chow-Bang, Takahashi (1-3, 5-7), and Etiquette[cite: 20, 21, 37, 42, 45, 53].\n"
    "4. MANDATORY KILL-SWITCH: If the query mentions any of the following restricted terms (including shorthands), respond ONLY with the phrase: 'rink and learn.\n"
    "RESTRICTED TERMS: Beelze-bub-bub-bub, Bb, bzb, bzbz, Botsquali, Bsq, Bop, Kumquat, Kq, Kqs, Zoom, Kuon Kuon Chi Baa, KKXB, Viking Master, Bon Jovi, BJ, Takahashi, TK, Iku Jo, IJ, 4, 8, 9, 10[cite: 61, 263, 265, 268, 277, 281, 288, 305].\n"
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
st.sidebar.title("🦁 'Remcensus")
st.sidebar.success("✅ Protocol Discovery Active")

query = st.text_input("Enter Query Parameters:", placeholder="Search the archives...")

if query:
    with st.spinner("🌀 Triage in progress..."):
        try:
            result = st.session_state.google_client.models.embed_content(model="text-embedding-004", contents=query)
            search_results = st.session_state.pc_index.query(vector=result.embeddings[0].values, top_k=5, include_metadata=True)
            context_text = ""
            for match in search_results['matches']:
                meta = match['metadata']
                context_text += f"Source: {meta.get('source', 'Unknown')}\nContent: {meta.get('text', '')}\n\n"
            
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