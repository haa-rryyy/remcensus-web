import streamlit as st
import google.generativeai as genai
from groq import Groq
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

# --- 3. MAIN INTERFACE ---
st.sidebar.title("🦁 'Remcensus")
ACTIVE_MODEL = "llama-3.3-70b-versatile"
st.sidebar.caption(f"Protocol: Groq/{ACTIVE_MODEL}")

query = st.text_input("Enter Query Parameters:", placeholder="e.g., Define 'a move'...")

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
            
            # REFINED ONTOLOGICAL PROMPT
            chat_completion = st.session_state.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the Librarian of the 'Remier League. "
                            "MANDATE: You must never didactically teach the rules of 'Rem. "
                            "DEFINITION OF TEACHING: Explaining how a move is executed, what its specific effect is, "
                            "how variations alter the gameplay sequence, or which vocalizations are required. "
                            "SAFE HARBOR: You ARE permitted to define the abstract nature of concepts. "
                            "For example: Defining 'a move' as a fundamental building block of a round is ALLOWED. "
                            "Explaining 'how to perform a move' or 'what happens when a move is played' is FORBIDDEN. "
                            "If you cross into procedural mechanics, you must respond ONLY with: 'rink and learn. "
                            "Focus on the 'what it is' (identity/history) and ignore the 'how it works' (procedure)."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context_text}\n\nQuestion: {query}"
                    }
                ],
                model=ACTIVE_MODEL,
            )
            
            final_answer = enforce_rem_lexicon(chat_completion.choices[0].message.content)
        except Exception as e:
            final_answer = f"⚠️ System Error: {e}"
            search_results = {'matches': []}

    st.info(final_answer)