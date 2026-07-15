import sys
from unittest.mock import MagicMock
# Mock VertexAI module to bypass Ragas internal import error
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

import os
import pandas as pd
from typing import List, Dict, Any, Tuple
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.document_manager import load_manifest
from backend.pipeline import query_pipeline
import backend.state as state

# Default evaluation datasets mapping by filename
FILE_TEST_QUESTIONS = {
    "HDFC-Surgicare-Plan-101N043V01.pdf": [
        {
            "question": "What is the name of policy holder?",
            "ground_truth": "Mrs. Ashwini Hiralal Rathod"
        },
        {
            "question": "What Option is chosen for surgical benefit?",
            "ground_truth": "Option A: Surgical Benefit with Hospitalisation Cash"
        },
        {
            "question": "What is the term of the policy?",
            "ground_truth": "20 years"
        },
        {
            "question": "What is the date of commencement of policy?",
            "ground_truth": "01/01/2009"
        }
    ],
    "DXC_Annual_Financial_Report_Sample.pdf": [
        {
            "question": "What was the Revenue of DXC in USD Billion for FY2025?",
            "ground_truth": "14.6"
        },
        {
            "question": "What was the Net Income in USD Billion for FY2024?",
            "ground_truth": "0.54"
        },
        {
            "question": "What is the strategic priority objective for Artificial Intelligence?",
            "ground_truth": "Accelerate enterprise AI adoption"
        }
    ],
    "wipo_pub_rn2021_18e-1-15.pdf": [
        {
            "question": "What accounting standards are used for WIPO financial statements?",
            "ground_truth": "International Public Sector Accounting Standards (IPSAS)"
        },
        {
            "question": "For which year ended is the WIPO Assemblies financial report submitted?",
            "ground_truth": "December 31, 2020"
        },
        {
            "question": "Which regulation of the WIPO Financial Regulations and Rules requires submitting the statements?",
            "ground_truth": "Regulation 6.7"
        }
    ]
}

def get_default_questions() -> List[Dict[str, str]]:
    """
    Returns default test questions matching currently ingested files in the manifest.
    If no recognized files are found, returns the combined questions of all active files,
    or falls back to HDFC questions if database is completely empty.
    """
    manifest = load_manifest()
    if not manifest:
        return FILE_TEST_QUESTIONS["HDFC-Surgicare-Plan-101N043V01.pdf"]
        
    combined_questions = []
    for filename in manifest.keys():
        if filename in FILE_TEST_QUESTIONS:
            combined_questions.extend(FILE_TEST_QUESTIONS[filename])
            
    if not combined_questions:
        # Fallback to HDFC if none matched
        return FILE_TEST_QUESTIONS["HDFC-Surgicare-Plan-101N043V01.pdf"]
        
    return combined_questions

DEFAULT_TEST_QUESTIONS = FILE_TEST_QUESTIONS["HDFC-Surgicare-Plan-101N043V01.pdf"]

def run_ragas_evaluation(
    llm_provider: str = "Groq",
    llm_key: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
    test_cases: List[Dict[str, str]] = None,
    ground_truth_style: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Runs Ragas evaluation on the current RAG pipeline using the selected LLM.
    Returns a tuple (results_df, average_scores_dict).
    """
    if not llm_key:
        raise ValueError("API Key is required to run Ragas evaluation LLM calls.")
        
    manifest = load_manifest()
    if not manifest:
        raise ValueError("No documents are currently indexed in the vector database. Please index a PDF first.")
        
    if test_cases is None:
        test_cases = DEFAULT_TEST_QUESTIONS
        
    state.initialize_models()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    # 1. Generate answers and retrieve contexts from our RAG pipeline
    for case in test_cases:
        q = case["question"]
        gt = case["ground_truth"]
        
        # Query pipeline (bypass cache to get fresh generation and retrieval)
        res = query_pipeline(
            query=q,
            retriever_type="hybrid",
            k=4,
            use_reranking=True,
            llm_provider=llm_provider,
            llm_key=llm_key,
            llm_model=llm_model,
            use_cache=False,
            ground_truth_style=ground_truth_style
        )
        
        questions.append(q)
        answers.append(res["answer"])
        contexts.append([chunk["content"] for chunk in res["retrieved_chunks"]])
        ground_truths.append(gt)
        
    # 2. Package as Hugging Face Dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # 3. Setup Ragas LLM and Embeddings wrappers
    # Add high timeout to LLM instances to prevent client-side network cuts
    if llm_provider == "Google Gemini":
        llm_instance = ChatGoogleGenerativeAI(model=llm_model, google_api_key=llm_key, temperature=0.0, timeout=120.0)
    elif llm_provider == "Groq":
        llm_instance = ChatGroq(model=llm_model, groq_api_key=llm_key, temperature=0.0, timeout=120.0)
    else:  # OpenAI
        llm_instance = ChatOpenAI(model=llm_model, openai_api_key=llm_key, temperature=0.0, timeout=120.0)
        
    ragas_llm = LangchainLLMWrapper(llm_instance)
    
    # Use our BGE embedding model for answer relevancy metric
    ragas_emb = LangchainEmbeddingsWrapper(state.embeddings)
    
    # Bind custom evaluator models to Ragas metrics
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_recall.llm = ragas_llm
    context_precision.llm = ragas_llm
    
    # 4. Execute evaluation with a custom RunConfig
    # Lowering max_workers prevents concurrent request spikes that trigger API rate limits and timeouts
    from ragas.run_config import RunConfig
    run_config = RunConfig(timeout=180, max_workers=2, max_retries=10)
    
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_config
    )
    
    # Convert result to pandas DataFrame for display
    results_df = evaluation_result.to_pandas()
    
    # Calculate average scores
    avg_scores = {
        "faithfulness": float(results_df["faithfulness"].mean()) if "faithfulness" in results_df else 0.0,
        "answer_relevancy": float(results_df["answer_relevancy"].mean()) if "answer_relevancy" in results_df else 0.0,
        "context_recall": float(results_df["context_recall"].mean()) if "context_recall" in results_df else 0.0,
        "context_precision": float(results_df["context_precision"].mean()) if "context_precision" in results_df else 0.0,
    }
    
    return results_df, avg_scores
