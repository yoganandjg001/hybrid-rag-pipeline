import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_huggingface import HuggingFaceCrossEncoder
except ImportError:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from backend.cache import SemanticCache
from backend.document_manager import load_manifest, character_based_split, structure_based_split

# Global variables defining the pipeline state
embeddings = None
vectordb = None
bm25_retriever = None
hybrid_retriever = None
mmr_retriever = None
cross_encoder = None
reranker = None
semantic_cache = None
all_chunks = []

def initialize_models(persist_dir: str = "chroma_db"):
    """Downloads and prepares embeddings, cross-encoder, and loads existing Chroma index on startup."""
    print("Initializing models and retrievers...")
    global embeddings, vectordb, cross_encoder, reranker, semantic_cache
    
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        semantic_cache = SemanticCache(embeddings)
    if cross_encoder is None:
        cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
        
    if vectordb is None:
        manifest = load_manifest()
        if os.path.exists(persist_dir) and manifest:
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            rebuild_from_persistent_files()
        else:
            vectordb = None

def rebuild_from_persistent_files():
    """Reloads BM25 and hybrid retrievers by reading stored files from documents/ folder."""
    print("Rebuilding in-memory retrievers from persistent files...")
    global all_chunks, bm25_retriever, hybrid_retriever, mmr_retriever, vectordb
    all_chunks = []
    doc_dir = "documents"
    manifest = load_manifest()
    
    if os.path.exists(doc_dir):
        pdf_files = [f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf")]
        for f in pdf_files:
            file_path = os.path.join(doc_dir, f)
            try:
                loader = PyPDFLoader(file_path=file_path)
                docs = loader.load()
                
                # Check manifest for chunking strategy and sizes
                info = manifest.get(f, {})
                strategy = info.get("chunking_strategy", "structure")
                chunk_size = info.get("chunk_size", 500 if strategy == "structure" else 500)
                chunk_overlap = info.get("chunk_overlap", 100)
                
                if strategy == "character":
                    chunks = character_based_split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                else:
                    chunks = structure_based_split(docs, max_chunk_chars=chunk_size)
                    
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error reloading persistent file {f}: {e}")
                
    if all_chunks and vectordb is not None:
        print(f"Rebuilding BM25 and hybrid retrievers with {len(all_chunks)} total chunks...")
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = 5
        # Initialize hybrid retriever with BM25 and vector retrievers
        # vector retriever uses the same Chroma vector database with embeddings
        vector_retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        print(f"BM25 and vector retrievers initialized using EnsembleRetriever. Hybrid weights: BM25=0.4, Vector=0.6")
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        mmr_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    else:
        bm25_retriever = None
        hybrid_retriever = None
        mmr_retriever = None
