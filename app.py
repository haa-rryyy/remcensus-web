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

# --- HELPER: ROBUST MODEL SELECTOR ---
def find_working_model():
    """Diagnoses available models and picks a working one."""
    try:
        # Get raw list from Google
        all_models = list(genai.list_models())
        model_names = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        # Store for debug display
        st.session_state.available_models = model_names
        
        # 1. Preferred Free Models (in order)
        # We strip 'models/' prefix just in case the API is being picky
        preferences = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-002",
            "gemini-pro",
            "gemini-1.0-pro"
        ]
        
        for pref in preferences:
            # Check for exact match or "models/" match
            if pref in model_names or f"models/{pref}" in model_names:
                return pref
        
        # 2. Fallback: Take the first one that has "flash" in it
        for m in model_names:
            if "flash" in m: return m
            
        # 3. Last Resort: Take the first one available
        if model_names: return model_names[0]
        
        return "gemini-1.5-flash" # Blind guess if list is empty
        
    except Exception as e:
        st.session_state.model_error = str(e)
        return "gemini-1.5-flash"

# --- 2. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.title("🧬 'Remcensus")
    st.caption("Registry of the 'Remier League")
    st.markdown("---")
    st.success("✅ System Online")
    st.info("Drink and Learn")
    
    st.markdown("---")
    st.markdown("**🔍 Model Diagnostics:**")
    
    active_model = find_working_model()
    st.code(f"Selected:\n{active_model}")
    
    # Show what was actually found (for debugging)
    if 'available_models' in st.session_state:
        with st.expander("See All Available Models"):
            st.write(st.session_state.available_models)

# --- 3. MAIN INTERFACE ---
st.markdown("## 🗃️ Archives of the 'Remier League")
st.markdown("Accessing archival data from 2017-Present.")

query = st.text_input("Enter Query Parameters:", placeholder="e.g., Who won the league in 2022?")

if query:
    with st.spinner("🔍 Querying Neural Database..."):
        try:
            # 1. Embed
            result = genai.embed_content(model="models/text-embedding-004", content=query)
            
            # 2. Search
            search_results = st.session_state.pc_index.query(
                vector=result['embedding'], top_k=5, include_metadata=True
            )

            # 3. Context
            context_text = ""
            for match in search_results['matches']:
                meta = match['metadata']
                context_text += f"Source: {meta.get('source', 'Unknown')}\nContent: {meta.get('text', '')}\n\n"

            # 4. Generate
            prompt = f"""
            You are the Librarian of the 'Remier League. Answer based strictly on context.
            Context: {context_text}
            Question: {query}
            """
            
            # CRITICAL FIX: Ensure the model name is clean
            clean_model_name = active_model.replace("models/", "")
            model = genai.GenerativeModel(clean_model_name)
            
            response = model.generate_content(prompt)
            final_answer = response.text

        except Exception as e:
            # Enhanced Error Reporting
            final_answer = f"⚠️ System Error: {e}"
            if "404" in str(e):
                final_answer += f"\n\n**Diagnosis:** The model '{active_model}' was rejected. Please check the sidebar to see which models are actually valid."
            search_results = {'matches': []}

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