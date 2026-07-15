import numpy as np
from langchain.retrievers import ContextualCompressionRetriever
import backend.state as state
from backend.retrieval import retrieve

def retrieve_and_rerank(query: str, retriever_type: str = "hybrid", k: int = 5, hybrid_weight: float = 0.5):
    """
    Reranks candidate documents retrieved using the specified strategy.
    Implements a larger candidate pool for high recall, uses the requested
    ContextualCompressionRetriever logic, and calculates explainability scores.
    """
    # Retrieve a larger pool to ensure target documents are captured
    print(f"Retrieving candidates for reranking with query: {query} and retriever type: {retriever_type}")
    candidate_k = max(k * 3, 12)
    print(f"Candidate pool size for reranking: {candidate_k} and hybrid weight: {hybrid_weight} and k: {k}")
    candidates = retrieve(query, retriever_type=retriever_type, k=candidate_k, hybrid_weight=hybrid_weight)
    if not candidates:
        return []
    
    # Contextual compression retriever matching user's reference code
    print(f"Reranking {len(candidates)} candidates using CrossEncoderReranker...")
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=state.reranker,
        base_retriever=state.vectordb.as_retriever(search_kwargs={"k": len(candidates)}),
    )
    
    # Rerank against the already-fetched candidate pool rather than re-querying the whole index.
    state.reranker.top_n = k
    scored = state.reranker.compress_documents(candidates, query)
    
    # Calculate cross encoder scores manually for explainability scores in UI
    pairs = [[query, doc.page_content] for doc in scored]
    scores = state.cross_encoder.score(pairs)
    for doc, score in zip(scored, scores):
        # Sigmoid normalization: 1 / (1 + exp(-x))
        normalized = 1.0 / (1.0 + np.exp(-float(score)))
        doc.metadata["relevance_score"] = normalized
        
    return scored
