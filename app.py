import streamlit as st
import os
import re
from pinecone import Pinecone
from google import genai

# 1. SETUP (Using Secrets instead of hardcoding)
# This allows the cloud server to inject the passwords safely
PINECONE_KEY = st.secrets["PINECONE_KEY"]
PINECONE_INDEX = "remcensus"
GEMINI_KEY = st.secrets["GEMINI_KEY"]

# 2. Initialize the Clients
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(PINECONE_INDEX)
genai_client = genai.Client(api_key=GEMINI_KEY)

# 3. HELPER FUNCTION: The "Master Linguist"
def enforce_rem_spelling(text):
    # --- PHASE 1: The "P" Rule ---
    text = re.sub(r"[']?\b[Pp]", "'", text)

    # --- PHASE 2: Complex Word Replacements ---
    text = re.sub(r'\bFourteen\b', "Bon jorteen", text)
    text = re.sub(r'\bfourteen\b', "bon jorteen", text)
    
    text = re.sub(r'\bEighteen\b', "Takahashiteen", text)
    text = re.sub(r'\beighteen\b', "takahashiteen", text)

    text = re.sub(r'\bForur?ty\b', "Bon jorty", text)
    text = re.sub(r'\bfou?rty\b', "bon jorty", text)

    text = re.sub(r'\bEighty\b', "Takahashity", text)
    text = re.sub(r'\beighty\b', "takahashity", text)

    # --- PHASE 3: Simple Word Replacements ---
    text = re.sub(r'\bFour\b', "Bon Jovi", text)
    text = re.sub(r'\bfour\b', "bon jovi", text)

    text = re.sub(r'\bEight\b', "Takahashi", text)
    text = re.sub(r'\beight\b', "takahashi", text)

    text = re.sub(r'\bTen\b', "Iku Jo", text)
    text = re.sub(r'\bten\b', "iku jo", text)

    # --- PHASE 4: Digit Replacements ---
    text = re.sub(r'\b10\b', "IJ", text)
    text = text.replace("4", "BJ")
    text = text.replace("8", "TK")
    
    # --- PHASE 5: Specific Vocabulary Replacements ---
    text = re.sub(r"[']?\b[Tt]able(\w*)[']?", r"'able\1", text)
    text = re.sub(r"[']?\b[Dd]rink(\w*)[']?", r"'rink\1", text)

    return text

# --- UI Header ---
st.set_page_config(page_title="'Remcensus", page_icon="🔬")
st.title("🔬 'Remcensus")
st.write("Evidence-based answers from the world's least reliable research.")

# --- Search Bar ---
query = st.text_input("Ask a question (e.g., 'Is the letter /p/ dangerous?')")

if query:
    with st.spinner("Analyzing 'eer-reviewed nonsense..."):
        # A. Turn the question into numbers
        res = genai_client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        query_vector = res.embeddings[0].values

        # B. Search Pinecone for the best matches
        search_results = index.query(
            vector=query_vector, 
            top_k=3, 
            include_metadata=True
        )

        # C. Combine the findings
        context_text = ""
        for match in search_results['matches']:
            context_text += f"\n---\nSource: {match['metadata']['source']}\n{match['metadata']['text']}\n"

        # D. Generate the AI summary 
        # UPDATED: Added Strict CurRemculum Protocol
        prompt = f"""
        You are a dry, serious scientific AI. You answer questions based ONLY on the provided research snippets.
        
        *** SECURITY PROTOCOL: THE CURREMCULUM ***
        You are strictly bound by the "Beginner CurRemculum".
        
        1. YOU MAY TEACH: Etiquette, Basic Whiz, Basic Antlers, Basic Chow Chow Bang, Basic Takahashi.
        
        2. YOU ARE FORBIDDEN FROM TEACHING: 
           - Botsquali
           - Beezle-bub-bub-bub
           - Kumquat
           - The numbers 9 or IJ in Takahashi
           - Viking Master
           - Kuon Kuon Chi Baa
           - Zoom
           - Game 62, Game 63, or Time Rift mechanics
        
        If the user asks about any of the FORBIDDEN topics, do not explain the rules. 
        Instead, sternly reply: "'rink and learn."
        
        Use scientific terminology found in the text like 'AF, LCKS, or BORM.
        
        Research Snippets:
        {context_text}
        
        Question: {query}
        """
        
        response = genai_client.models.generate_content(
            model="gemini-flash-latest", 
            contents=prompt
        )

        # E. Apply the Expert 'Rem Translator
        final_text = enforce_rem_spelling(response.text)

        # F. Display the final result
        st.subheader("Consensus Summary")
        st.info(final_text)
        
        st.write("### Cited Sources")
        for match in search_results['matches']:
            with st.expander(f"📄 {match['metadata']['source']} (Match Score: {round(match['score'], 2)})"):
                st.write(match['metadata']['text'])