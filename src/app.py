import streamlit as st
from RAGPipeline  import RAGChromaPipeline

# --- Load RAG System ---
if "rag" not in st.session_state:
    st.session_state.rag = RAGChromaPipeline(persist_directory="./chroma_db")

st.set_page_config(page_title="CrediTrust Complaint Assistant", layout="wide")

st.title("💬 CrediTrust Complaint Assistant")
st.markdown("Ask any question about customer complaints across financial products.")
st.markdown("---")

# --- Chat Input ---
user_question = st.text_input("🔍 Ask a question", placeholder="e.g. Why are people unhappy with Buy Now, Pay Later?", key="input")

# --- Submit Button ---
if st.button("Ask"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            result = st.session_state.rag.run(user_question)

            # Display Answer
            st.subheader("📘 Answer")
            st.write(result["answer"])

            # Display Sources
            st.subheader("📄 Sources (from complaint chunks)")
            for i, src in enumerate(result["sources"][:3], start=1):
                st.markdown(f"**{i}.** *Product:* `{src.get('product')}` — *Complaint ID:* `{src.get('complaint_id')}`")
                st.code(src.get("page_content", "[No content available]"), language="markdown")

# --- Clear Button ---
if st.button("Clear"):
    st.session_state.input = ""
    st.experimental_rerun()
