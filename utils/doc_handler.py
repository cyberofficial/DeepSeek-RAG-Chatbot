import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.build_graph import build_knowledge_graph
from rank_bm25 import BM25Okapi
import os
import re


def process_documents(uploaded_files, reranker, embedding_model, base_url, chunk_size=1000, chunk_overlap=200, progress_bar=None, status_text=None):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True
    documents = []
    total_steps = 6  # Total number of major processing steps
    current_step = 0
    
    def update_progress(message, step_increment=1):
        nonlocal current_step
        if status_text:
            status_text.text(message)
        if progress_bar:
            # Ensure we don't exceed 100%
            current_step = min(current_step + step_increment, total_steps)
            progress_bar.progress(current_step / total_steps)
    
    # Create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    update_progress("Loading documents...")
    # Process files
    for idx, file in enumerate(uploaded_files):
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue
                
            documents.extend(loader.load())
            os.remove(file_path)
            if len(uploaded_files) > 1:
                update_progress(f"Loaded {idx + 1}/{len(uploaded_files)} documents", 0)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return

    update_progress("Splitting text into chunks...")
    # Text splitting
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    text_contents = [doc.page_content for doc in texts]

    update_progress("Generating embeddings...")
    # 🚀 Hybrid Retrieval Setup
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    
    update_progress("Creating vector store...")
    # Vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    
    update_progress("Setting up BM25 retrieval...")
    # BM25 store
    bm25_retriever = BM25Retriever.from_texts(
        text_contents,
        bm25_impl=BM25Okapi,
        preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split()
    )

    # Ensemble retrieval
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_store.as_retriever(search_kwargs={"k": 5})
        ],
        weights=[0.4, 0.6]
    )

    update_progress("Building knowledge graph...")
    # Store in session
    st.session_state.retrieval_pipeline = {
        "ensemble": ensemble_retriever,
        "reranker": reranker,
        "texts": text_contents,
        "knowledge_graph": build_knowledge_graph(texts)
    }

    st.session_state.documents_loaded = True
    st.session_state.processing = False

    update_progress("Processing complete!", 0)  # Don't increment counter for completion message
    
    # ✅ Knowledge Graph Stats
    if "knowledge_graph" in st.session_state.retrieval_pipeline:
        G = st.session_state.retrieval_pipeline["knowledge_graph"]
        with st.expander("📊 Knowledge Graph Statistics"):
            st.write(f"🔗 Total Nodes: {len(G.nodes)}")
            st.write(f"🔗 Total Edges: {len(G.edges)}")
            st.write(f"🔗 Sample Nodes: {list(G.nodes)[:10]}")
            st.write(f"🔗 Sample Edges: {list(G.edges)[:10]}")
