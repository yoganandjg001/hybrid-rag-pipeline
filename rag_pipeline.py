# Setup dynamic aliasing first for compatibility with this environment
import sys
try:
    import langchain_classic.retrievers
    import langchain_classic.retrievers.document_compressors
    sys.modules['langchain.retrievers'] = langchain_classic.retrievers
    sys.modules['langchain.retrievers.document_compressors'] = langchain_classic.retrievers.document_compressors
except Exception:
    pass

import backend.state as state
from backend.cache import SemanticCache
from backend.document_manager import (
    get_file_hash, load_manifest, save_manifest,
    character_based_split, structure_based_split,
    ingest_pdf, delete_document
)
from backend.state import initialize_models, rebuild_from_persistent_files
from backend.retrieval import retrieve
from backend.reranker import retrieve_and_rerank
from backend.llm import generate_answer
from backend.pipeline import query_pipeline

# Dynamic module-level attribute lookup and assignment to sync state variables
def __getattr__(name):
    if name in ('embeddings', 'vectordb', 'bm25_retriever', 'hybrid_retriever',
                'mmr_retriever', 'cross_encoder', 'reranker', 'semantic_cache',
                'all_chunks'):
        return getattr(state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __setattr__(name, value):
    if name in ('embeddings', 'vectordb', 'bm25_retriever', 'hybrid_retriever',
                'mmr_retriever', 'cross_encoder', 'reranker', 'semantic_cache',
                'all_chunks'):
        import backend.state as state
        setattr(state, name, value)
    else:
        globals()[name] = value
