import os
import streamlit as st
import requests


st.title("Custom RAG system for PDF Q&A")
API_URL = os.getenv("API_URL", "http://localhost:8000")


# File upload
st.header("PDF Uploader")
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=['pdf'],
    accept_multiple_files=True
)

if uploaded_files and st.button("Ingest Documents"):
    with st.spinner("Processing PDFs..."):
        files = [("files", (file.name, file, "application/pdf")) for file in uploaded_files]
        response = requests.post(f"{API_URL}/ingest", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Ingested {result['total_chunks']} chunks from {len(uploaded_files)} file(s)")
        
        else:
            st.error(f"Ingest Error: {response.text}")


# Query interface
st.header("Ask Questions")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching..."):
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query}
        )
        
        ## If we get a response from the API
        if response.status_code == 200:
            result = response.json()
            
            st.subheader("Answer:")
            st.write(result["answer"])
            
            if result["chunks"]:
                st.markdown("**Retrieved Chunks:**")
                for i, chunk in enumerate(result["chunks"]):
                    with st.expander(f"Chunk {i+1} (Score: {chunk['score']:.3f})"):
                        st.write(chunk["chunk"])
        
        else:
            st.error(f"Search Error: {response.text}")