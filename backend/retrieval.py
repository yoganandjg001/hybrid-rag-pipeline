import backend.state as state

# This module provides a unified interface for retrieving documents 
# based on a query using different retrieval strategies: similarity search, hybrid retrieval, and maximum marginal relevance (MMR) search. 
# It leverages the initialized models and retrievers from the state module to perform the retrieval operations.
# when a user query is provided using chatbot, the retrieve function determines the appropriate retrieval strategy based on the specified retriever_type and executes the corresponding retrieval method.
def retrieve(query: str, retriever_type: str = "hybrid", k: int = 5, hybrid_weight: float = 0.5, score_threshold: float = 0.35):
    """Retrieves candidates using similarity, hybrid, or MMR."""
    print(f"Retrieving documents for query: {query} using retriever type: {retriever_type} with k={k} and hybrid_weight={hybrid_weight}")
    
    if state.vectordb is None:
        return []
        
    # Prepend BAAI/bge-base-en-v1.5 instruction prefix for vector search queries
    vector_query = f"Represent this sentence for searching relevant passages: {query}"
    
    if retriever_type == "hybrid" and state.hybrid_retriever is not None:
        # Similarity search first; fall back to hybrid if the top match is weak.
        scored = state.vectordb.similarity_search_with_relevance_scores(vector_query, k=k)
        top_score = scored[0][1] if scored else 0.0
        
        if top_score >= score_threshold:
            print(f"Similarity search sufficient (top score={top_score:.2f})")
            return [doc for doc, _ in scored]
            
        print(f"Similarity search weak (top score={top_score:.2f}) -> falling back to hybrid retrieval")
        state.bm25_retriever.k = k
        if len(state.hybrid_retriever.retrievers) > 1:
            state.hybrid_retriever.retrievers[1].search_kwargs["k"] = k
        return state.hybrid_retriever.invoke(query)[:k]
        
    elif retriever_type == "mmr" and state.mmr_retriever is not None:
        return state.vectordb.max_marginal_relevance_search(vector_query, k=k, fetch_k=min(k * 4, 30))
        
    else:  # Standard similarity
        return state.vectordb.similarity_search(vector_query, k=k)
