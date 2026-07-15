from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def generate_answer(
    query: str, 
    docs_with_scores: List[Document], 
    provider: str, 
    api_key: str, 
    model_name: str,
    ground_truth_style: bool = False
) -> str:
    """Generates an answer using the selected LLM provider and API key."""
    if not api_key:
        return "⚠️ LLM response not generated. Please enter your API Key in the sidebar to get answers."
        
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')} | Section: {doc.metadata.get('section')}]\n{doc.page_content}"
        for doc in docs_with_scores
    )
    
    if ground_truth_style:
        prompt = ChatPromptTemplate.from_template(
            """You are an expert assistant. Answer the user's question strictly based on the provided context.
You must answer concisely and directly, returning ONLY the exact requested entity, name, date, or value (matching a clean database entry/ground truth format).
Do NOT include any conversational preamble, full sentences, or introductory text (e.g. do not say "The name of the policy holder is Mrs. Ashwini Hiralal Rathod", output exactly: "Mrs. Ashwini Hiralal Rathod").

If the context does not contain the answer, reply exactly: "I cannot find the answer in the provided document."
Do not use outside knowledge.

CONTEXT:
{context}

QUESTION: {query}
"""
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            """You are an expert assistant. Answer the user's question strictly based on the provided context.
If the context does not contain the answer, reply exactly: "I cannot find the answer in the provided document."
Do not use outside knowledge. Keep your answer professional, clear, and well-structured.

CONTEXT:
{context}

QUESTION: {query}
"""
        )

    try:
        if provider == "Google Gemini":
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)
        elif provider == "Groq":
            llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.0)
        else:  # OpenAI
            llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.0)
            
        chain = prompt | llm
        res = chain.invoke({"query": query, "context": context})
        return res.content
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"
