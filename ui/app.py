"""Streamlit chat UI for the agentic RAG pipeline.

Run with: streamlit run ui/app.py
"""

import os
import tempfile

import streamlit as st

from llm.groq_client import LLMError
from retrieval.ingestion import ingest_pdf
from workflow import rag_workflow

st.set_page_config(page_title="Agentic RAG", page_icon="📄")

st.title("📄 Agentic RAG")
st.caption(
    "Router → query rewrite → hybrid retrieval (BM25 + embeddings) → "
    "retrieval grading → self-correcting retries → cited answer"
)


def render_trace(trace):
    """Show what the agentic pipeline actually did for this query."""
    with st.expander("Pipeline trace"):
        st.markdown(f"**Routed to retrieval:** {trace['needs_retrieval']}")
        if trace["needs_retrieval"]:
            st.markdown(f"**Rewritten query:** {trace['rewritten_query']}")
            st.markdown(f"**Retries:** {trace['retry_count']}")
            if trace.get("grader_feedback"):
                st.markdown(f"**Grader feedback:** {trace['grader_feedback']}")
            for i, chunk in enumerate(trace["retrieved_chunks"], start=1):
                source = chunk.get("source", "unknown") if isinstance(chunk, dict) else "unknown"
                page = chunk.get("page", "?") if isinstance(chunk, dict) else "?"
                text = chunk["text"] if isinstance(chunk, dict) else str(chunk)
                st.markdown(f"**[{i}] {source}, page {page}**")
                st.text(text)


with st.sidebar:
    st.header("Ingest a document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded is not None and st.button("Ingest"):
        with st.spinner("Extracting, chunking, and embedding..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = os.path.join(tmp_dir, uploaded.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                count = ingest_pdf(pdf_path)
            # The workflow's store and answer cache still point at the old
            # corpus; reset both so the next query hits the new document.
            rag_workflow.store.refresh()
            rag_workflow.clear_workflow_cache()
        st.success(f"Ingested {count} chunks from {uploaded.name}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("trace"):
            render_trace(message["trace"])

query = st.chat_input("Ask a question about the ingested document")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Running agentic pipeline..."):
            try:
                trace = rag_workflow.run_workflow(query, return_trace=True)
            except LLMError as error:
                st.error(f"LLM call failed: {error}")
                st.stop()
        st.markdown(trace["answer"])
        render_trace(trace)

    st.session_state.messages.append(
        {"role": "assistant", "content": trace["answer"], "trace": trace}
    )
