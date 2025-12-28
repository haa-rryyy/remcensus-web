import streamlit as st
import google.generativeai as genai
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
        genai.configure(api_key=st.secrets["GEMINI_KEY"])
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

# --- 3. UNIVERSAL SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "You are the Librarian of the 'Remier League. Classify the user query into one of three Tiers and respond accordingly.\n\n"
    "TIER 1: PERMITTED (Didactic Teaching Allowed)\n"
    "Topics: Basic Whiz (Whiz, Bang, Bounce, Alley-oop), Basic Antlers, "
    "Basic Chow-Chow-Bang (Chow, Bang), Takahashi (Numbers 1-3, 5-7), "
    "Etiquette (Meeting, Chair, Timing, Vocalisation, Courts).\n"
    "Action: Explain clinically based on context.\n\n"
    "TIER 2: ONTOLOGICAL (Definitions Only)\n"
    "Topics: Abstract definitions of 'a move', 'a game', 'a court', or 'a variation'.\n"
    "Action: Define WHAT it is. Refuse HOW it works. If procedure is asked -> Tier 3.\n\n"
    "TIER 3: RESTRICTED (Strict Silence)\n"
    "Topics: \n"
    "- Whiz Moves: Botsquali (Bsq), Beelze-bub-bub-bub (Bb), Bop, Alpha.\n"
    "- Chow Moves: Kumquat (Kq), Kumquat support (Kqs).\n"
    "- Takahashi Numbers: Bon Jovi (4), Takahashi (8), Number 9, Iku Jo (10).\n"
    "- Games: Zoom, Kuon Kuon Chi Baa, Viking Master, Bon Jovi, Full vessel consumption.\n"
    "- Any named variation not in Tier 1.\n"
    "Action: Respond ONLY with the phrase: 'rink and learn."
)

# --- 4. TRIPLE-ENGINE HANDLER ---
def generate_response(context, query):
    # ATTEMPT 1: GROQ
    try:
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content, "Groq (Llama 3.3)"
    except Exception as e_groq:
        # ATTEMPT 2: GEMINI (Syntax Audited)
        try:
            model = genai.GenerativeModel(
                model_name="models/gemini-1.5-flash",
                system_instruction=SYSTEM_PROMPT
            )
            response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")
            return response.text, "Gemini (1.5 Flash)"
        except Exception as e_gem:
            # ATTEMPT 3: HUGGING FACE
            try:
                response = st.session_state.hf_client.chat_completion(
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content, "Hugging Face (Mistral-7B)"
            except Exception as e_hf:
                return f"⚠️ SYSTEM FAILURE: All protocols failed. Groq: {e_groq}. Gemini: {e_gem}. HF: {e_hf}.", "OFFLINE"

# --- 5. MAIN INTERFACE ---
st.sidebar.title("🦁 'Remcensus")
st.sidebar.success("✅ Triple-Engine Cascade Active")

query = st.text_input("Enter Query Parameters:", placeholder="Querying the archives...")

if query:
    with st.spinner("🌀 Whizzing..."):
        try:
            result = genai.embed_content(model="models/text-embedding-004", content=query)
            search_results = st.session_state.pc_index.query(vector=result['embedding'], top_k=5, include_metadata=True)
            context_text = ""
            for match in search_results['matches']:
                meta = match['metadata']
                context_text += f"Source: {meta.get('source', 'Unknown')}\nContent: {meta.get('text', '')}\n\n"
            
            raw_text, engine_used = generate_response(context_text, query)
            final_answer = enforce_rem_lexicon(raw_text)
            
            st.caption(f"Generated via: {engine_used}")
            st.info(final_answer)
        except Exception as e:
            st.error(f"⚠️ Critical Error: {e}")