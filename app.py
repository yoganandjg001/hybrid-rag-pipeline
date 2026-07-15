import os
import time
import streamlit as st
import pandas as pd
import rag_pipeline
import eval_rag

# Page Config
st.set_page_config(
    page_title="DXC RAG Diagnostics Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Premium Design & Fonts
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Apply modern typography */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Styled Header */
.main-header {
    background: linear-gradient(135deg, #8A2387 0%, #E94057 50%, #F27121 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
    text-align: center;
}

.sub-header {
    color: #8A94A6;
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 2rem;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}

.metric-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: #FF7B00;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.9rem;
    color: #8A94A6;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Passage Retrieval Cards */
.chunk-card {
    background: rgba(255, 255, 255, 0.03);
    border-left: 6px solid #FF5E3A;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0 16px 16px 0;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.chunk-card.mmr {
    border-left-color: #2EC4B6;
}

.chunk-card.hybrid {
    border-left-color: #7209B7;
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.8rem;
    font-weight: 600;
}

.chunk-source {
    color: #FF9E00;
    font-size: 0.95rem;
}

.chunk-score {
    background: rgba(255, 94, 58, 0.15);
    color: #FF7B54;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
}

.chunk-score.mmr {
    background: rgba(46, 196, 182, 0.15);
    color: #2EC4B6;
}

.chunk-score.hybrid {
    background: rgba(114, 9, 183, 0.15);
    color: #B5179E;
}

.chunk-text {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #E2E8F0;
    margin-bottom: 0.8rem;
    white-space: pre-line;
}

.chunk-metadata {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    font-size: 0.8rem;
    color: #8A94A6;
    margin-top: 0.8rem;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding-top: 0.6rem;
}

.chunk-meta-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.doc-info-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0px 0px;
    gap: 8px;
    font-size: 1rem;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<div class='main-header'>🔮 DXC RAG Pipeline & Diagnostics</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Explainable Hybrid Retrieval & Cross-Encoder Re-Ranking Playground</div>", unsafe_allow_html=True)

# ----------------- SESSION STATE & MODEL INITS -----------------

@st.cache_resource
def load_and_init_models():
    """Download and prepare HF Embeddings & Cross-Encoder models once."""
    with st.spinner("🚀 Downloading and loading HuggingFace models (only once, first run may take a minute)..."):
        rag_pipeline.initialize_models()
    return True

models_ready = load_and_init_models()

# Load manifest details
manifest = rag_pipeline.load_manifest()
total_chunks_count = len(rag_pipeline.all_chunks)

# Default Ingestion: Auto-index HDFC PDF on first run if nothing is in the manifest
if not manifest:
    default_pdf = "HDFC-Surgicare-Plan-101N043V01.pdf"
    if os.path.exists(default_pdf):
        with st.spinner("⏳ Automatically indexing workspace document HDFC-Surgicare..."):
            status, fname = rag_pipeline.ingest_pdf(default_pdf)
            if status == "success":
                st.sidebar.success(f"Indexed workspace document ({fname})")
                st.rerun()

# ----------------- SIDEBAR CONFIGURATION -----------------

st.sidebar.markdown("## ⚙️ RAG Configuration")

# 1. LLM API Key & Model Config
st.sidebar.markdown("### 🔑 LLM Provider Setup")
llm_provider = st.sidebar.selectbox("Select LLM Provider", ["Google Gemini", "OpenAI", "Groq"], index=2)

# Retrieve keys from environment if set
default_gemini = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
default_openai = os.environ.get("OPENAI_API_KEY", "")
default_groq = os.environ.get("GROQ_API_KEY", "")

if llm_provider == "Google Gemini":
    llm_key = st.sidebar.text_input(
        "Gemini API Key",
        value=default_gemini,
        type="password",
        help="Input Gemini API key to enable LLM response generation."
    )
    llm_model = st.sidebar.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        index=0
    )
elif llm_provider == "Groq":
    llm_key = st.sidebar.text_input(
        "Groq API Key",
        value=default_groq,
        type="password",
        help="Input Groq API key to enable LLM response generation."
    )
    llm_model = st.sidebar.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )
else:
    llm_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=default_openai,
        type="password",
        help="Input OpenAI API key to enable LLM response generation."
    )
    llm_model = st.sidebar.selectbox(
        "OpenAI Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )

# 2. Retrieval Parameters
st.sidebar.markdown("### 🔍 Retrieval Parameters")
retriever_type = st.sidebar.selectbox(
    "Primary Retriever Strategy",
    ["hybrid", "mmr", "vector"],
    format_func=lambda x: {
        "hybrid": "Hybrid (BM25 + Dense Vector)",
        "mmr": "MMR (Maximal Marginal Relevance)",
        "vector": "Standard Vector Similarity"
    }[x],
    index=0
)

k_val = st.sidebar.slider("Number of retrieved chunks (K)", min_value=1, max_value=10, value=4)

if retriever_type == "hybrid":
    hybrid_weight = st.sidebar.slider(
        "Hybrid Weight (Dense vs Sparse)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        help="0.0 = Pure Keyword (BM25); 1.0 = Pure Semantic (Vector)"
    )
else:
    hybrid_weight = 0.5

use_rerank = st.sidebar.checkbox("Enable Cross-Encoder Re-Ranking", value=True)
use_cache = st.sidebar.checkbox("Enable Semantic Query Cache", value=True)
ground_truth_style = st.sidebar.checkbox("Strict Ground Truth Response", value=False, help="Forces the LLM to output concise, direct answers matching ground truth format (no conversational preamble).")

if st.sidebar.button("🧹 Clear Semantic Cache"):
    if rag_pipeline.semantic_cache:
        rag_pipeline.semantic_cache.entries = []
        st.sidebar.success("Cache cleared!")

if st.sidebar.button("💬 Clear Chat History"):
    st.session_state.messages = []
    st.sidebar.success("Chat history cleared!")
    st.rerun()

# ----------------- MAIN LAYOUT -----------------

# Refresh manifest states
manifest = rag_pipeline.load_manifest()

# Metrics Header Row
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-label'>📂 Indexed Documents</div>
        <div style='font-size: 1.1rem; font-weight: 700; color: #E2E8F0; margin: 0.8rem 0;'>
            {len(manifest)} Active files
        </div>
    </div>
    """, unsafe_allow_html=True)
with m_col2:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-label'>🧩 Total Chunks</div>
        <div class='metric-number'>{total_chunks_count}</div>
    </div>
    """, unsafe_allow_html=True)
with m_col3:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-label'>🏷️ Embedding Model</div>
        <div style='font-size: 1.1rem; font-weight: 700; color: #FF7B00; margin: 0.8rem 0;'>bge-base-en-v1.5</div>
    </div>
    """, unsafe_allow_html=True)
with m_col4:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-label'>🔄 Reranker</div>
        <div style='font-size: 1.1rem; font-weight: 700; color: #2EC4B6; margin: 0.8rem 0;'>
            {'Contextual Compressor' if use_rerank else 'Disabled'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Q&A Playground", "🔬 Retrieval Diagnostics", "📁 Document Manager & Inspector", "📊 Ragas Evaluator"])

with tab1:
    st.markdown("### 💬 Ask the Document")
    
    # Initialize message list in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "diagnostics" in msg:
                diag = msg["diagnostics"]
                badge_class = "hybrid" if diag.get("retriever_type") == "hybrid" else ("mmr" if diag.get("retriever_type") == "mmr" else "")
                with st.expander("🔍 View Retrieval Diagnostics & Sources"):
                    st.markdown(f"⏱️ **Retrieval Time:** {diag['retrieval_time']*1000:.1f} ms")
                    if diag.get("cache_hit"):
                        st.info(f"⚡ **Semantic Cache Hit!** Similar past query matched with **{diag.get('cache_similarity', 0)*100:.1f}%** cosine similarity.")
                    
                    st.markdown("##### 📖 Sources & Passages:")
                    for idx, chunk in enumerate(diag["retrieved_chunks"]):
                        score_label = "Re-ranked Score" if use_rerank else "Similarity Score"
                        st.markdown(f"""
                        <div class='chunk-card {badge_class}' style='margin-bottom:0.8rem; padding: 0.8rem;'>
                            <div class='chunk-header' style='font-size:0.85rem; margin-bottom:0.4rem;'>
                                <span class='chunk-source' style='font-size:0.85rem;'>📍 Passage #{idx+1} — {chunk['section']}</span>
                                <span class='chunk-score {badge_class}' style='font-size:0.75rem; padding:0.1rem 0.5rem;'>{score_label}: {chunk['score']:.4f}</span>
                            </div>
                            <div class='chunk-text' style='font-size: 0.85rem; line-height: 1.4; color: #CBD5E1;'>{chunk['content']}</div>
                            <div class='chunk-metadata' style='font-size: 0.75rem; margin-top:0.4rem; padding-top:0.4rem;'>
                                <div class='chunk-meta-item'>📄 <b>Source:</b> {chunk['source']} | Page: {chunk['page']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
    # Chat input
    user_query = st.chat_input("Ask a question about the indexed document(s)...")
    
    if user_query:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Ingestion guard
        if not manifest:
            with st.chat_message("assistant"):
                st.error("Please ingest a document first in the 'Document Manager' tab.")
            st.session_state.messages.append({"role": "assistant", "content": "❌ Please ingest a document first."})
        else:
            with st.spinner("🧠 Querying RAG Pipeline..."):
                res = rag_pipeline.query_pipeline(
                    query=user_query,
                    retriever_type=retriever_type,
                    k=k_val,
                    hybrid_weight=hybrid_weight,
                    use_reranking=use_rerank,
                    llm_provider=llm_provider,
                    llm_key=llm_key,
                    llm_model=llm_model,
                    use_cache=use_cache,
                    ground_truth_style=ground_truth_style
                )
                
            # Append assistant response
            with st.chat_message("assistant"):
                st.markdown(res["answer"])
                diag_data = {
                    "retriever_type": retriever_type,
                    "retrieval_time": res["retrieval_time"],
                    "cache_hit": res.get("cache_hit", False),
                    "cache_similarity": res.get("cache_similarity", 0.0),
                    "retrieved_chunks": res["retrieved_chunks"]
                }
                
                badge_class = "hybrid" if retriever_type == "hybrid" else ("mmr" if retriever_type == "mmr" else "")
                with st.expander("🔍 View Retrieval Diagnostics & Sources"):
                    st.markdown(f"⏱️ **Retrieval Time:** {res['retrieval_time']*1000:.1f} ms")
                    if res.get("cache_hit", False):
                        st.info(f"⚡ **Semantic Cache Hit!** Similar past query matched with **{res.get('cache_similarity', 0)*100:.1f}%** cosine similarity.")
                    
                    st.markdown("##### 📖 Sources & Passages:")
                    for idx, chunk in enumerate(res["retrieved_chunks"]):
                        score_label = "Re-ranked Score" if use_rerank else "Similarity Score"
                        st.markdown(f"""
                        <div class='chunk-card {badge_class}' style='margin-bottom:0.8rem; padding: 0.8rem;'>
                            <div class='chunk-header' style='font-size:0.85rem; margin-bottom:0.4rem;'>
                                <span class='chunk-source' style='font-size:0.85rem;'>📍 Passage #{idx+1} — {chunk['section']}</span>
                                <span class='chunk-score {badge_class}' style='font-size:0.75rem; padding:0.1rem 0.5rem;'>{score_label}: {chunk['score']:.4f}</span>
                            </div>
                            <div class='chunk-text' style='font-size: 0.85rem; line-height: 1.4; color: #CBD5E1;'>{chunk['content']}</div>
                            <div class='chunk-metadata' style='font-size: 0.75rem; margin-top:0.4rem; padding-top:0.4rem;'>
                                <div class='chunk-meta-item'>📄 <b>Source:</b> {chunk['source']} | Page: {chunk['page']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "diagnostics": diag_data
            })
            st.rerun()

with tab2:
    st.markdown("### 🔬 Retrieval Comparison & Accuracy Diagnostics")
    st.markdown("""
    Compare how different retrieval configurations alter chunk rankings and confidence scores. 
    Running comparisons below helps debug vector representation and BM25 weighting.
    """)
    
    # Fetch last user query in session state for comparison defaults
    default_compare = "What are the death benefits?"
    if "messages" in st.session_state and st.session_state.messages:
        user_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        if user_queries:
            default_compare = user_queries[-1]
            
    compare_query = st.text_input("Enter comparison test query:", value=default_compare)
    
    if st.button("Run Comparison Diagnostic"):
        if not manifest:
            st.error("Please ingest a document first.")
        else:
            with st.spinner("Comparing retrieval strategies..."):
                # 1. Pure Vector (Similarity)
                t_vec = time.time()
                vec_docs = rag_pipeline.retrieve(compare_query, retriever_type="vector", k=k_val)
                vec_time = time.time() - t_vec
                
                # 2. Hybrid (Ensemble)
                t_hyb = time.time()
                hybrid_docs = rag_pipeline.retrieve(compare_query, retriever_type="hybrid", k=k_val, hybrid_weight=hybrid_weight)
                hybrid_time = time.time() - t_hyb
                
                # 3. MMR
                t_mmr = time.time()
                mmr_docs = rag_pipeline.retrieve(compare_query, retriever_type="mmr", k=k_val)
                mmr_time = time.time() - t_mmr
                
                # Re-rank scores for each
                vec_ranked = rag_pipeline.retrieve_and_rerank(compare_query, retriever_type="vector", k=k_val)
                hybrid_ranked = rag_pipeline.retrieve_and_rerank(compare_query, retriever_type="hybrid", k=k_val, hybrid_weight=hybrid_weight)
                mmr_ranked = rag_pipeline.retrieve_and_rerank(compare_query, retriever_type="mmr", k=k_val)
                
            # Table visualization
            st.markdown("#### 📊 Metric Comparisons")
            
            top_vec_score = vec_ranked[0].metadata.get("relevance_score", 0.0) if vec_ranked else 0.0
            top_hyb_score = hybrid_ranked[0].metadata.get("relevance_score", 0.0) if hybrid_ranked else 0.0
            top_mmr_score = mmr_ranked[0].metadata.get("relevance_score", 0.0) if mmr_ranked else 0.0
            
            # Handle float lists
            if isinstance(top_vec_score, list): top_vec_score = float(top_vec_score[0])
            if isinstance(top_hyb_score, list): top_hyb_score = float(top_hyb_score[0])
            if isinstance(top_mmr_score, list): top_mmr_score = float(top_mmr_score[0])
            
            avg_vec = sum(float(d.metadata.get("relevance_score", 0.0)[0] if isinstance(d.metadata.get("relevance_score", 0.0), list) else d.metadata.get("relevance_score", 0.0)) for d in vec_ranked) / len(vec_ranked) if vec_ranked else 0.0
            avg_hyb = sum(float(d.metadata.get("relevance_score", 0.0)[0] if isinstance(d.metadata.get("relevance_score", 0.0), list) else d.metadata.get("relevance_score", 0.0)) for d in hybrid_ranked) / len(hybrid_ranked) if hybrid_ranked else 0.0
            avg_mmr = sum(float(d.metadata.get("relevance_score", 0.0)[0] if isinstance(d.metadata.get("relevance_score", 0.0), list) else d.metadata.get("relevance_score", 0.0)) for d in mmr_ranked) / len(mmr_ranked) if mmr_ranked else 0.0

            metrics_df = pd.DataFrame({
                "Strategy": ["Standard Vector", "Hybrid (BM25 + Vector)", "MMR (Diversity)"],
                "Retrieval Time": [f"{vec_time*1000:.2f} ms", f"{hybrid_time*1000:.2f} ms", f"{mmr_time*1000:.2f} ms"],
                "Top Chunk Score": [f"{top_vec_score:.4f}", f"{top_hyb_score:.4f}", f"{top_mmr_score:.4f}"],
                "Avg Chunk Score": [f"{avg_vec:.4f}", f"{avg_hyb:.4f}", f"{avg_mmr:.4f}"]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visual ranking comparison
            st.markdown("#### 🔍 Ranked Chunks Comparison")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader("Standard Vector")
                for idx, doc in enumerate(vec_ranked):
                    score = doc.metadata.get("relevance_score", 0.0)
                    if isinstance(score, list): score = float(score[0])
                    st.markdown(f"""
                    <div class='chunk-card'>
                        <div class='chunk-header'>
                            <span class='chunk-source'>Rank #{idx+1} — Page {doc.metadata.get('page')}</span>
                            <span class='chunk-score'>{score:.4f}</span>
                        </div>
                        <div class='chunk-text' style='font-size:0.85rem;'>{doc.page_content[:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with c2:
                st.subheader("Hybrid (BM25 + Vector)")
                for idx, doc in enumerate(hybrid_ranked):
                    score = doc.metadata.get("relevance_score", 0.0)
                    if isinstance(score, list): score = float(score[0])
                    st.markdown(f"""
                    <div class='chunk-card hybrid'>
                        <div class='chunk-header'>
                            <span class='chunk-source'>Rank #{idx+1} — Page {doc.metadata.get('page')}</span>
                            <span class='chunk-score hybrid'>{score:.4f}</span>
                        </div>
                        <div class='chunk-text' style='font-size:0.85rem;'>{doc.page_content[:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with c3:
                st.subheader("MMR (Diversity)")
                for idx, doc in enumerate(mmr_ranked):
                    score = doc.metadata.get("relevance_score", 0.0)
                    if isinstance(score, list): score = float(score[0])
                    st.markdown(f"""
                    <div class='chunk-card mmr'>
                        <div class='chunk-header'>
                            <span class='chunk-source'>Rank #{idx+1} — Page {doc.metadata.get('page')}</span>
                            <span class='chunk-score mmr'>{score:.4f}</span>
                        </div>
                        <div class='chunk-text' style='font-size:0.85rem;'>{doc.page_content[:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 📁 Document Manager & Index Explorer")
    st.markdown("""
    Ingest, delete, and inspect files incrementally in the RAG pipeline.
    The system performs automatic duplicate checksum checks to prevent redundant indexing.
    """)
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 📤 Ingest New Document")
        
        chunking_strategy = st.selectbox(
            "Select Chunking Strategy",
            ["structure", "character"],
            format_func=lambda x: "Structure-Based (Logical Sections)" if x == "structure" else "Character-Based (Recursive Character Splitter)",
            index=0,
            help="Structure-based parses headings and splits along document sections. Character-based splits text into fixed-length segments."
        )
        
        # Display strategy defaults and sliders
        if chunking_strategy == "structure":
            st.markdown("""
            🤖 **Application Default**: Max section chunk size of **1500 characters**.
            """)
            custom_chunk_size = st.slider(
                "Max Section Chunk Size (chars)", 
                min_value=200, 
                max_value=2500, 
                value=1500, 
                step=50,
                help="Maximum characters in a section chunk before sub-splitting."
            )
            custom_overlap = 0 # Not applicable for structure split
        else:
            st.markdown("""
            🤖 **Application Default**: Chunk size of **500 characters** with **100 characters** overlap.
            """)
            custom_chunk_size = st.slider(
                "Chunk Size (chars)", 
                min_value=100, 
                max_value=1500, 
                value=500, 
                step=50,
                help="Size of individual text chunks."
            )
            custom_overlap = st.slider(
                "Chunk Overlap (chars)", 
                min_value=0, 
                max_value=300, 
                value=100, 
                step=10,
                help="Overlapping characters between adjacent chunks."
            )
        
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
            
        uploaded_file = st.file_uploader(
            "Upload PDF file to index", 
            type=["pdf"],
            key=f"pdf_uploader_{st.session_state.uploader_key}"
        )
        
        if uploaded_file is not None:
            # Save temp file for hash calculation & potential ingestion
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Perform duplicate content check before running full parser
            file_hash = rag_pipeline.get_file_hash(temp_path)
            duplicate_detected = False
            collision_detected = False
            duplicate_name = ""
            
            for name, info in manifest.items():
                if info.get("hash") == file_hash:
                    duplicate_detected = True
                    duplicate_name = name
                    break
                elif name == uploaded_file.name:
                    collision_detected = True
                    break
                    
            if duplicate_detected:
                st.error(f"❌ **Duplicate Content Detected!** This file matches the exact contents of the already ingested document: `{duplicate_name}`.")
                st.info("Ingestion skipped to save indexing costs and prevent database noise.")
                
            elif collision_detected:
                st.warning(f"⚠️ **Filename Collision!** A file named `{uploaded_file.name}` already exists but has different contents.")
                overwrite = st.button("Overwrite Existing Document", type="primary")
                if overwrite:
                    with st.spinner("⏳ Overwriting document..."):
                        status, name = rag_pipeline.ingest_pdf(
                            temp_path, 
                            chunking_strategy=chunking_strategy, 
                            chunk_size=custom_chunk_size,
                            chunk_overlap=custom_overlap,
                            force=True
                        )
                        if status == "success":
                            st.success(f"✅ Successfully updated and re-indexed `{name}`.")
                            time.sleep(1)
                            st.session_state.uploader_key += 1 # Reset uploader widget
                            st.rerun()
            else:
                ingest_btn = st.button("Start Incremental Ingestion", type="primary")
                if ingest_btn:
                    with st.spinner("⏳ Splitting sections & indexing document..."):
                        status, name = rag_pipeline.ingest_pdf(
                            temp_path, 
                            chunking_strategy=chunking_strategy,
                            chunk_size=custom_chunk_size,
                            chunk_overlap=custom_overlap
                        )
                        if status == "success":
                            st.success(f"✅ Successfully ingested `{name}` into Chroma vector DB.")
                            time.sleep(1)
                            st.session_state.uploader_key += 1 # Reset uploader widget
                            st.rerun()
                            
    with col_right:
        st.markdown("#### 📋 Ingested Documents List")
        if not manifest:
            st.info("No documents currently ingested in Chroma DB.")
        else:
            for name, info in manifest.items():
                with st.container():
                    strategy = info.get('chunking_strategy', 'structure')
                    if strategy == "structure":
                        strategy_label = "Structure-Based (Logical Sections)"
                        chunk_size_label = "Max 1500 characters limit per section split"
                    else:
                        strategy_label = "Character-Based (Recursive Splitter)"
                        chunk_size_label = "Fixed 500 characters (100 characters overlap)"
                        
                    st.markdown(f"""
                    <div class='doc-info-card'>
                        <div style='font-size: 1.05rem; font-weight: 700; color: #FF7B00; margin-bottom:0.3rem;'>📄 {name}</div>
                        <div style='font-size: 0.85rem; color:#8A94A6; line-height:1.5;'>
                            📅 <b>Added:</b> {info['added_at']}<br/>
                            🧩 <b>Chunks Created:</b> {info['chunk_count']}<br/>
                            ⚙️ <b>Strategy Used:</b> {strategy_label}<br/>
                            📏 <b>Chunk Size Config:</b> {chunk_size_label}<br/>
                            🔒 <b>MD5 Checksum:</b> <span style='font-family:monospace; font-size:0.75rem;'>{info['hash']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Delete button
                    del_btn = st.button(f"🗑️ Delete `{name}`", key=f"del_{name}")
                    if del_btn:
                        with st.spinner("⏳ Deleting document chunks and rebuilding indexes..."):
                            success = rag_pipeline.delete_document(name)
                            if success:
                                st.success(f"Deleted `{name}` successfully.")
                                time.sleep(1)
                                st.rerun()
                                
    # Document explorer list (for exploring sections)
    st.markdown("---")
    st.markdown("#### 📂 Explore Index Segments & Sections")
    if not rag_pipeline.all_chunks:
        st.info("No chunks indexed to explore.")
    else:
        # Group chunks by section for viewer
        sections = {}
        for idx, chunk in enumerate(rag_pipeline.all_chunks):
            sec_name = f"{chunk.metadata.get('source', 'Unknown')} ➡️ {chunk.metadata.get('section', 'General')}"
            if sec_name not in sections:
                sections[sec_name] = []
            sections[sec_name].append((idx, chunk))
            
        for sec_name, chunk_list in sections.items():
            with st.expander(f"📍 {sec_name} ({len(chunk_list)} chunk{'s' if len(chunk_list) > 1 else ''})"):
                for idx, chunk in chunk_list:
                    st.markdown(f"**Chunk #{idx+1} (Page {chunk.metadata.get('page')}, Length: {len(chunk.page_content)} characters):**")
                    st.code(chunk.page_content, language="text")
                    st.markdown("---")

with tab4:
    st.markdown("### 📊 Pipeline Accuracy Evaluation (Ragas)")
    st.markdown("""
    Evaluate the overall performance of the current RAG pipeline (Retrieval and Generation) using **Ragas** (Retrieval Augmented Generation Assessment).
    The system measures faithfulness, answer relevancy, context recall, and context precision.
    """)
    
    # Check uploader status
    if not manifest:
        st.info("Please index a document under the 'Document Manager & Inspector' tab to run an evaluation.")
    else:
        # Choose test suite / ground truths
        test_suite = st.selectbox(
            "Select Evaluation Test Suite (Ground Truths)",
            [
                "Auto-detect from Ingested Documents",
                "HDFC Surgicare Plan Questions",
                "DXC Financial Report Questions",
                "WIPO Financial Report Questions",
                "Upload Custom CSV"
            ],
            index=0,
            help="Choose the dataset of questions and expected ground truths to evaluate against."
        )
        
        # Load questions based on selection
        if "prev_test_suite" not in st.session_state or st.session_state.prev_test_suite != test_suite:
            st.session_state.prev_test_suite = test_suite
            if test_suite == "Auto-detect from Ingested Documents":
                st.session_state.eval_questions = list(eval_rag.get_default_questions())
            elif test_suite == "HDFC Surgicare Plan Questions":
                st.session_state.eval_questions = list(eval_rag.FILE_TEST_QUESTIONS["HDFC-Surgicare-Plan-101N043V01.pdf"])
            elif test_suite == "DXC Financial Report Questions":
                st.session_state.eval_questions = list(eval_rag.FILE_TEST_QUESTIONS["DXC_Annual_Financial_Report_Sample.pdf"])
            elif test_suite == "WIPO Financial Report Questions":
                st.session_state.eval_questions = list(eval_rag.FILE_TEST_QUESTIONS["wipo_pub_rn2021_18e-1-15.pdf"])
            elif test_suite == "Upload Custom CSV":
                st.session_state.eval_questions = [] # Will be populated by file uploader
        
        # Custom CSV Uploader block
        if test_suite == "Upload Custom CSV":
            uploaded_eval_file = st.file_uploader("Upload CSV containing 'question' and 'ground_truth' columns", type=["csv"])
            if uploaded_eval_file is not None:
                try:
                    df_eval = pd.read_csv(uploaded_eval_file)
                    if "question" in df_eval.columns and "ground_truth" in df_eval.columns:
                        st.session_state.eval_questions = df_eval[["question", "ground_truth"]].to_dict(orient="records")
                        st.success(f"Successfully loaded {len(st.session_state.eval_questions)} test cases from CSV!")
                    else:
                        st.error("Error: CSV must contain both 'question' and 'ground_truth' columns.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
                    
        st.markdown("#### 📋 Test Cases Dataset")
        # Display editable table of test questions and ground truths
        edited_questions = []
        for idx, case in enumerate(st.session_state.eval_questions):
            col_q, col_gt = st.columns([1, 1])
            with col_q:
                q_val = st.text_input(f"Question #{idx+1}", value=case["question"], key=f"eval_q_{idx}")
            with col_gt:
                gt_val = st.text_input(f"Ground Truth #{idx+1}", value=case["ground_truth"], key=f"eval_gt_{idx}")
            edited_questions.append({"question": q_val, "ground_truth": gt_val})
            
        st.session_state.eval_questions = edited_questions
        
        # Ingestion and running control
        if not llm_key:
            st.warning("⚠️ Please configure your API Key in the sidebar to run the Ragas evaluator.")
        else:
            if st.button("🚀 Run Ragas Evaluation", type="primary"):
                with st.spinner("⏳ Querying pipeline and running Ragas evaluation LLM grading (takes ~30-60s)..."):
                    try:
                        results_df, avg_scores = eval_rag.run_ragas_evaluation(
                            llm_provider=llm_provider,
                            llm_key=llm_key,
                            llm_model=llm_model,
                            test_cases=st.session_state.eval_questions,
                            ground_truth_style=ground_truth_style
                        )
                        
                        st.success("✅ Ragas Evaluation Completed Successfully!")
                        
                        # Display metrics cards
                        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                        with col_e1:
                            st.metric("Faithfulness", f"{avg_scores['faithfulness']*100:.1f}%", help="Is the answer grounded in context?")
                        with col_e2:
                            st.metric("Answer Relevancy", f"{avg_scores['answer_relevancy']*100:.1f}%", help="Does the answer address the question?")
                        with col_e3:
                            st.metric("Context Recall", f"{avg_scores['context_recall']*100:.1f}%", help="Did we retrieve all info needed?")
                        with col_e4:
                            st.metric("Context Precision", f"{avg_scores['context_precision']*100:.1f}%", help="Are relevant chunks ranked first?")
                            
                        # Map Ragas standardized column names back to user-friendly names dynamically
                        rename_map = {
                            "user_input": "question",
                            "response": "answer",
                            "reference": "ground_truth",
                            "retrieved_contexts": "contexts"
                        }
                        rename_map = {k: v for k, v in rename_map.items() if k in results_df.columns}
                        if rename_map:
                            results_df = results_df.rename(columns=rename_map)
                            
                        # Safely select columns that exist in the dataframe to display
                        display_cols = [
                            c for c in ["question", "ground_truth", "answer", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
                            if c in results_df.columns
                        ]
                        
                        st.markdown("#### 📊 Detailed Row-by-Row Scores")
                        st.dataframe(
                            results_df[display_cols],
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Evaluation failed: {str(e)}")
