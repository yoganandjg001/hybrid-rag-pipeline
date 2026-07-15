import time
from typing import Dict, Any
import backend.state as state
from backend.retrieval import retrieve
from backend.reranker import retrieve_and_rerank
from backend.llm import generate_answer

def query_pipeline(
    query: str, 
    retriever_type: str = "hybrid", 
    k: int = 5, 
    hybrid_weight: float = 0.5, 
    use_reranking: bool = True, 
    llm_provider: str = "Groq", 
    llm_key: str = "", 
    llm_model: str = "llama-3.3-70b-versatile",
    use_cache: bool = True,
    ground_truth_style: bool = False
) -> Dict[str, Any]:
    state.initialize_models()
    
    # 1. Semantic Cache Lookup
    if use_cache and state.semantic_cache:
        print(f"Checking semantic cache for query: {query}")
        cached = state.semantic_cache.get(query)
        if cached is not None:
            print(f"Cache hit for query: {query}")
            res, cache_score = cached
            result = dict(res)
            result["cache_hit"] = True
            result["cache_similarity"] = cache_score
            return result

    # 2. Retrieval & Re-ranking
    print(f"Retrieving documents for query: {query}")
    t_start = time.time()
    if use_reranking:
        print(f"Using reranking with retriever type: {retriever_type}")
        ranked_docs = retrieve_and_rerank(query, retriever_type=retriever_type, k=k, hybrid_weight=hybrid_weight)
    else:
        print(f"Using direct retrieval with retriever type: {retriever_type}")
        ranked_docs = retrieve(query, retriever_type=retriever_type, k=k, hybrid_weight=hybrid_weight)
    t_end = time.time()
    retrieve_time = t_end - t_start

    if not ranked_docs:
        print(f"No documents retrieved for query: {query}")
        return {
            "answer": "No context found. Please ingest documents first.",
            "retrieved_chunks": [],
            "retrieval_time": retrieve_time,
            "cache_hit": False
        }

    # 3. LLM Generation
    answer = generate_answer(
        query=query,
        docs_with_scores=ranked_docs,
        provider=llm_provider,
        api_key=llm_key,
        model_name=llm_model,
        ground_truth_style=ground_truth_style
    )

    # 4. Format outputs
    chunks_info = []
    for doc in ranked_docs:
        score = doc.metadata.get("relevance_score", 1.0)
        if isinstance(score, list):
            score = float(score[0])
        chunks_info.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "section": doc.metadata.get("section", "N/A"),
            "score": float(score)
        })

    result = {
        "answer": answer,
        "retrieved_chunks": chunks_info,
        "retrieval_time": retrieve_time,
        "cache_hit": False
    }
    
    # Save to cache if successful
    if use_cache and llm_key and not answer.startswith("⚠️") and not answer.startswith("❌") and state.semantic_cache:
        print(f"Saving query to semantic cache: {query}")
        state.semantic_cache.set(query, result)

    return result
