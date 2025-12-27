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

# API KEYS
PINECONE_KEY = "pcsk_5Z6WFV_7oQoBHq752tQWBHA2VWdRPA7cYWCZgHyVeLBATX4iVnd899XA6N7XgeTRLFTCDK"
PINECONE_INDEX = "remcensus"
GEMINI_KEY = "AIzaSyBF7yZAfy-na9pO52yfGQIhnqpFNNsvRjM"

# Initialize Connections
if 'init_done' not in st.session_state:
    pc = Pinecone(api_key=PINECONE_KEY)
    st.session_state.pc_index = pc.Index(PINECONE_INDEX)
    genai.configure(api_key=GEMINI_KEY)
    st.session_state.init_done = True

# --- HELPER: SAFE MODEL SELECTOR ---
def get_safe_model():
    """Finds a free-tier compatible model, avoiding the 'Pro' quota traps."""
    try:
        # 1. Try to list available models
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 2. STRICT PRIORITY LIST (Free Tier Favorites)
        # We explicitly look for these names because we know they work.
        priorities = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.0-pro",
            "models/gemini-pro"
        ]
        
        for priority in priorities:
            if priority in available_models:
                return priority
                
        # 3. Fallback: If exact matches fail, take the first 'flash' model found
        for m in available_models:
            if 'flash' in m: return m
            
        # 4. Desperation Fallback
        return "models/gemini-1.5-flash"
        
    except Exception as e:
        # If listing fails, force the standard one
        return "models/gemini-1.5-flash"

# --- 2. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.title("🧬 'Remcensus")
    st.caption("Registry of the 'Remier League")
    st.markdown("---")
    
    st.markdown("**User Status:**")
    st.success("✅ Authorized (Guest)")
    
    st.markdown("**System Protocol:**")
    st.info("Drink and Learn")
    
    st.markdown("---")
    
    # Get the safe model
    active_model = get_safe_model()
    st.text(f" Active Model:\n {active_model.replace('models/', '')}")

# --- 3. MAIN INTERFACE ---
st.markdown("## 🗃️ Archives of the 'Remier League")
st.markdown("Accessing archival data from 2017-Present.")

# Search Bar
query = st.text_input("Enter Query Parameters:", placeholder="e.g., Who won the league in 2022?")

if query:
    # --- SEARCH LOGIC ---
    with st.spinner("🔍 Querying Neural Database..."):
        try:
            # 1. Embed the query
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query
            )
            query_embedding = result['embedding']

            # 2. Search Pinecone
            search_results = st.session_state.pc_index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )

            # 3. Construct Context
            context_text = ""
            for match in search_results['matches']:
                meta = match['metadata']
                source = meta.get('source', 'Unknown')
                text = meta.get('text', '')
                context_text += f"Source: {source}\nContent: {text}\n\n"

            # 4. Generate Answer
            prompt = f"""
            You are the Librarian of the 'Remier League. Answer the question based strictly on the context below.
            If the answer is not in the context, say "Data not found in the archives."
            Keep the tone clinical, precise, and slightly bureaucratic.
            
            Context:
            {context_text}
            
            Question: {query}
            """
            
            # Use the safe model we found earlier
            model = genai.GenerativeModel(active_model)
            response = model.generate_content(prompt)
            final_answer = response.text

        except Exception as e:
            # Clean Error Handling
            if "429" in str(e):
                final_answer = "⚠️ Error: Quota Limit Exceeded. Please wait 60 seconds."
            elif "404" in str(e):
                final_answer = f"⚠️ Error: Model {active_model} unavailable. Try reloading."
            else:
                final_answer = f"⚠️ System Error: {e}"
            search_results = {'matches': []}

    # --- 4. DISPLAY RESULTS ---
    col1, col2 = st.columns([2, 1]) 

    with col1:
        st.subheader("📝 Consensus Summary")
        if "⚠️" in final_answer:
            st.error(final_answer)
        else:
            st.info(final_answer)

    with col2:
        st.subheader("📂 Reference Data")
        if search_results['matches']:
            st.caption(f"Found {len(search_results['matches'])} relevant records.")
            for match in search_results['matches']:
                meta = match['metadata']
                with st.expander(f"📄 {meta.get('source', 'Doc')}"):
                    score = round(match['score'] * 100, 1)
                    st.caption(f"Relevance Score: {score}%")
                    st.markdown(f"*{meta.get('text', '')[:300]}...*")
        else:
            st.write("No references found.")