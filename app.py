import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="'Remcensus Registry",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SECURE CONNECTION (No hardcoded keys!)
if 'init_done' not in st.session_state:
    try:
        # Load keys from Streamlit Secrets
        pc = Pinecone(api_key=st.secrets["PINECONE_KEY"])
        st.session_state.pc_index = pc.Index(st.secrets["PINECONE_INDEX"])
        genai.configure(api_key=st.secrets["GEMINI_KEY"])
        st.session_state.init_done = True
    except Exception as e:
        st.error(f"🔐 Security Error: Keys missing from Streamlit Secrets. {e}")
        st.stop()

# --- HELPER: SAFE MODEL SELECTOR ---
def get_safe_model():
    """Finds a free-tier compatible model."""
    try:
        # 1. Ask Google what is available
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 2. Priority List (Free Tier)
        priorities = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-001",
            "models/gemini-pro"
        ]
        
        for priority in priorities:
            if priority in available_models:
                return priority
        
        # 3. Fallback
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

# --- 2. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.title("🧬 'Remcensus")
    st.caption("Registry of the 'Remier League")
    st.markdown("---")
    st.success("✅ System Online")
    st.info("Drink and Learn")
    
    # Debug info (optional)
    active_model = get_safe_model()
    st.caption(f"Model: {active_model.replace('models/', '')}")

# --- 3. MAIN INTERFACE ---
st.markdown("## 🗃️ Archives of the 'Remier League")
st.markdown("Accessing archival data from 2017-Present.")

# Search Bar
query = st.text_input("Enter Query Parameters:", placeholder="e.g., Who won the league in 2022?")

if query:
    with st.spinner("🔍 Querying Neural Database..."):
        try:
            # 1. Embed Query
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query
            )
            
            # 2. Search Pinecone
            search_results = st.session_state.pc_index.query(
                vector=result['embedding'],
                top_k=5,
                include_metadata=True
            )

            # 3. Build Context
            context_text = ""
            for match in search_results['matches']:
                meta = match['metadata']
                context_text += f"Source: {meta.get('source', 'Unknown')}\nContent: {meta.get('text', '')}\n\n"

            # 4. Generate Answer
            prompt = f"""
            You are the Librarian of the 'Remier League. Answer strictly based on the context below.
            If the answer is missing, state "Data not found in the archives."
            Tone: Clinical, precise, bureaucratic.
            
            Context:
            {context_text}
            
            Question: {query}
            """
            
            model = genai.GenerativeModel(active_model)
            response = model.generate_content(prompt)
            final_answer = response.text

        except Exception as e:
            final_answer = f"⚠️ System Error: {e}"
            search_results = {'matches': []}

    # --- 4. DISPLAY ---
    col1, col2 = st.columns([2, 1]) 

    with col1:
        st.subheader("📝 Consensus Summary")
        if "⚠️" in final_answer: st.error(final_answer)
        else: st.info(final_answer)

    with col2:
        st.subheader("📂 Reference Data")
        if search_results['matches']:
            for match in search_results['matches']:
                with st.expander(f"📄 {match['metadata'].get('source', 'Doc')}"):
                    st.caption(f"Score: {round(match['score'] * 100, 1)}%")
                    st.write(match['metadata'].get('text', '')[:300] + "...")
        else:
            st.write("No references found.")