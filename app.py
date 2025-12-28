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
        (r"\b4\b", "BJ"), (r"\b8\b", "TK"),
        (r"\b10\b", "IJ")
    ]
    for pattern, sub in num_map:
        text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    return text

# --- 3. UNIVERSAL SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "You are the Librarian of the 'Remier League. Classify the user query into one of three Tiers.\n\n"
    "TIER 1: PERMITTED\n"
    "Topics: Basic Whiz (Whiz, Bang, Bounce, Alley-oop), Basic Antlers, "
    "Basic Chow-Chow-Bang (Chow, Bang), Takahashi (Numbers 1-3, 5-7), "
    [cite_start]"Etiquette (Meeting, Chair, Timing, Vocalisation, Courts)[cite: 20, 21, 37, 42, 45, 53].\n"
    [cite_start]"Action: Explain clinically[cite: 19].\n\n"
    "TIER 2: ONTOLOGICAL\n"
    "Topics: Abstract definitions of 'a move', 'a game', or 'a court'.\n"
    "Action: Define WHAT it is. Refuse HOW it works.\n\n"
    "TIER 3: RESTRICTED\n"
    "Topics: Botsquali, Beelze-bub-bub-bub, Bop, Kumquat, Zoom, Kuon Kuon Chi Baa, "
    [cite_start]"Viking Master, Bon Jovi, Full vessel consumption, numbers 4, 8, 9, 10[cite: 40, 41, 52, 57, 58, 62, 63, 64, 65, 66, 263, 265, 268, 277, 281, 288, 298, 305].\n"
    [cite_start]"Action: Respond ONLY with: 'rink and learn[cite: 13, 14, 18]."
)

# --- 4. STABILIZED TRIPLE-ENGINE HANDLER ---
def generate_response(context, query):
    # PRIMARY: GROQ
    try:
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content, "Groq (Llama 3.3)"
    except Exception as e_groq:
        # SECONDARY: GEMINI (Corrected path)
        try:
            model = genai.GenerativeModel("models/gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)
            response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")
            return response.text, "Gemini (1.5 Flash)"
        except Exception as e_gem:
            # TERTIARY: HUGGING FACE (Chat-native model)
            try:
                response = st.session_state.hf_client.chat_completion(
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}],
                    max_tokens=500
                )
                return response.choices[0].message.content, "Hugging Face (Mistral-7B)"
            except Exception as e_hf:
                return f"⚠️ SYSTEM FAILURE: All protocols failed.\nGroq: {e_groq}\nGemini: {e_gem}\nHF: {e_hf}", "OFFLINE"

# --- 5. MAIN INTERFACE ---
st.sidebar.title("🦁 'Remcensus")
st.sidebar.success("✅ Triple-Engine Cascade Active")

query = st.text_input("Enter Query Parameters:", placeholder="e.g., Query the archives...")

if query:
    with st.spinner("🌀 Whizzing (checking protocols)..."):
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