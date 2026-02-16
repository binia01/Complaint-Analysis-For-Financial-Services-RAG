"""CrediTrust Streamlit chat application."""

import os
import time

import streamlit as st

from src.rag_engine import RAGPipeline

# --- CONFIGURATION ---
st.set_page_config(
    page_title="CrediTrust AI Analyst", page_icon="🏦", layout="wide"
)

# --- HEADER & SIDEBAR ---
st.title("🏦 CrediTrust Intelligent Complaint Analysis")

with st.sidebar:
    st.header("Control Panel")

    # 1. Try to get from Environment
    api_key = os.environ.get("GOOGLE_API_KEY")

    # 2. If not in Env, try Streamlit Secrets
    if not api_key and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]

    # 3. If still missing, ask User
    if not api_key:
        api_key = st.text_input("🔑 Enter Google API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key  # Set it for the session
    else:
        st.success("API Key Detected ✅")

    st.markdown("---")

    if st.button("🗑️ Reset Conversation", type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built for CrediTrust Financial Services")

# --- STOP IF NO KEY ---
if not os.environ.get("GOOGLE_API_KEY"):
    st.warning(
        "⚠️ Please enter your Google API Key in the sidebar to continue."
    )
    st.stop()


# --- INITIALIZE RAG ENGINE ---
@st.cache_resource
def load_rag_engine() -> RAGPipeline:
    return RAGPipeline(
        parquet_path="./data/complaint_embeddings.parquet",
        vector_db_path="./chroma_db_full",
    )


try:
    with st.spinner("Initializing Knowledge Base..."):
        rag = load_rag_engine()
except Exception as e:
    st.error(f"Failed to load RAG Pipeline. Error: {e}")
    st.stop()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("🔍 View Source Evidence"):
                for source in message["sources"]:
                    st.info(f"**Date:** {source['date']}\n\n{source['text']}")

if prompt := st.chat_input("Ask a question about customer complaints..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Analyzing complaints..."):
            try:
                result = rag.answer_question(prompt)
                answer_text = result["answer"]
                source_docs = result["source_documents"]
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.stop()

        for chunk in answer_text.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        formatted_sources = []
        with st.expander("🔍 View Source Evidence (Verified Claims)"):
            for i, doc in enumerate(source_docs):
                meta_date = doc.metadata.get("date", "Unknown")
                content = doc.page_content
                st.markdown(f"**Source {i+1}** (Date: {meta_date})")
                st.info(content)
                formatted_sources.append({"date": meta_date, "text": content})

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": formatted_sources,
    })
