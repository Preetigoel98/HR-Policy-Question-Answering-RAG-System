# Streamlit HR Assistant App

import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

# Page config
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🤖",
    layout="wide"
)


# Load model and database (cached)
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="hr_policies")
    return model, collection


model, collection = load_resources()

# Header
st.title("🤖 HR Policy Assistant")
st.markdown("Ask questions about company policies and get instant answers!")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📚 About")
    st.write("This RAG-powered assistant searches through:")
    st.write("- Employee Handbook")
    st.write("- WFH Policy")
    st.write("- Expense Reimbursement")
    st.write("- HR Policies")

    st.markdown("---")
    st.header("⚙️ Settings")
    top_k = st.slider("Number of results", 1, 5, 3)

# Main search
question = st.text_input("💬 Ask your question:", placeholder="e.g., What is the work from home policy?")

if st.button("🔍 Search", type="primary") or question:
    if question:
        with st.spinner("Searching policies..."):
            # Search
            query_embedding = model.encode([question])[0]
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Display results
            st.success(f"Found {len(results['documents'][0])} relevant sections!")

            for i in range(len(results['documents'][0])):
                source = results['metadatas'][0][i]['source']
                content = results['documents'][0][i]
                score = 1 - results['distances'][0][i]

                with st.expander(f"📄 Result {i + 1}: {source} (Relevance: {score:.2%})"):
                    st.markdown(f"**Source:** `{source}`")
                    st.markdown(f"**Relevance Score:** {score:.2%}")
                    st.markdown("**Content:**")
                    st.write(content)
    else:
        st.warning("Please enter a question!")

# Example questions
st.markdown("---")
st.subheader("💡 Example Questions:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Work from home policy?"):
        st.session_state.example_q = "What is the work from home policy?"
        st.rerun()

with col2:
    if st.button("💰 Expense claims?"):
        st.session_state.example_q = "How do I claim expenses?"
        st.rerun()

with col3:
    if st.button("📅 Leave policy?"):
        st.session_state.example_q = "What is the leave policy?"
        st.rerun()

# Handle example questions
if 'example_q' in st.session_state:
    question = st.session_state.example_q
    del st.session_state.example_q

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Powered by RAG (Retrieval-Augmented Generation)")