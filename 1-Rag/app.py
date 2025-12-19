import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import re

# Import the optimized agent creator
from optimized_agent import create_agent
from performance_optimizer import display_performance_stats

def link_citations(text):
    """
    Automatically link legal citations to Indian Kanoon search.
    Matches:
    - Section 123
    - Article 14
    - Order 39 Rule 1
    - 2023 SCC 123 (Case citations)
    """
    # Base URL for Indian Kanoon search
    base_url = "https://indiankanoon.org/search/?formInput="
    
    # Regex patterns for common legal citations
    patterns = [
        (r'(Section\s+\d+[A-Z]?)', r'\1'),
        (r'(Article\s+\d+[A-Z]?)', r'\1'),
        (r'(Order\s+\d+\s+Rule\s+\d+)', r'\1'),
        (r'(\d{4}\s+SCC\s+\d+)', r'\1')
    ]
    
    linked_text = text
    for pattern, group in patterns:
        # Replace with markdown link
        # We use a lambda to URL encode the match
        def replace_match(match):
            citation = match.group(0)
            encoded_citation = citation.replace(" ", "+")
            return f"[{citation}]({base_url}{encoded_citation})"
            
        linked_text = re.sub(pattern, replace_match, linked_text)
        
    return linked_text

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

## If you do not have open AI key use the below Huggingface embedding
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

from langchain_huggingface import HuggingFaceEmbeddings

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize vector store if it exists on disk
if "vector_store" not in st.session_state:
    faiss_store_path = "faiss_store"
    if os.path.exists(faiss_store_path):
        try:
            # Load fine-tuned embeddings model
            st.session_state.embeddings_model = HuggingFaceEmbeddings(
                model_name="models/kanoon_embedder",
                model_kwargs={"device": "cpu"}
            )
            # Load existing vector store
            st.session_state.vector_store = FAISS.load_local(
                faiss_store_path, st.session_state.embeddings_model, allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"Could not load existing vector store: {e}")
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# We don't need to init LLM here for the agent, the agent handles it, 
# but we might keep it for other purposes if needed. 
# For now, we'll rely on the agent.

# --- Page Config ---
st.set_page_config(
    page_title="Legal Research Engine", 
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
        font-weight: 700;
    }
    
    h1 {
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2.5rem;
    }
    
    h2 {
        color: #374151 !important;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.8rem;
    }
    
    h3 {
        color: #4b5563 !important;
        margin: 1rem 0 0.8rem 0;
        font-size: 1.4rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Answer card styling */
    .answer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        color: white !important;
    }
    
    .answer-card h3 {
        color: white !important;
        margin-bottom: 16px;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .answer-content {
        background: rgba(255,255,255,0.95);
        padding: 20px;
        border-radius: 12px;
        color: #1f2937 !important; /* Force dark text */
        line-height: 1.6;
        font-size: 1rem;
    }

    .answer-content strong {
        color: #111827 !important; /* Force darker bold text */
        font-weight: 700;
    }
    
    .time-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-left: 12px;
        color: white !important;
    }
    
    /* Source card styling */
    .source-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        margin: 12px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .source-card:hover {
        background: #f1f5f9;
        transform: translateX(4px);
        transition: all 0.3s ease;
    }
    
    .source-title {
        color: #3b82f6 !important;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 8px;
    }
    
    .source-query {
        color: #64748b !important;
        font-style: italic;
        font-size: 0.9rem;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white !important;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 24px 0 16px 0;
        font-weight: 600;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: transparent;
    }

    /* User message specific styling to ensure visibility */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }

    /* Force text color in expanders */
    .streamlit-expanderContent {
        color: #1f2937 !important;
        background-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Title with better styling
st.markdown("<h1>⚖️ Legal AI Research Engine</h1>", unsafe_allow_html=True)

# Top buttons with better alignment
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    show_knowledge_base = st.button("📚 Knowledge Base", use_container_width=True)
with col2:
    show_system_status = st.button("⚙️ System Status", use_container_width=True)
with col3:
    show_performance = st.button("📊 Performance", use_container_width=True)
with col4:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.divider()

# Show sections based on button clicks
if show_knowledge_base:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("📚 Knowledge Base")
    st.write("Upload PDF files to create your custom research index. The system will process your documents and create embeddings for semantic search.")
    uploaded_files = st.file_uploader("Drop PDF files here", type=["pdf"], accept_multiple_files=True)
    if st.button("🔄 Update Knowledge Base"):
        with st.spinner("Processing documents..."):
            create_vector_embedding()
        st.success("✅ Knowledge base updated successfully!")
    st.markdown('</div>', unsafe_allow_html=True)

if show_system_status:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("⚙️ System Status")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists("faiss_store"):
            st.success("✓ Vector Database Active")
        else:
            st.warning("⚠ Database Not Initialized")
    
    with col_b:
        st.info("🚀 System Ready")
    
    st.divider()
    st.markdown("**Powered by:**")
    st.write("• **Groq** - Fast LLM inference")
    st.write("• **LangGraph** - Agent orchestration")
    st.write("• **Indian Kanoon** - Legal database access")
    st.markdown('</div>', unsafe_allow_html=True)

if show_performance:
    display_performance_stats()

st.divider()

# --- Function to create embeddings and vector store (lazy) ---
def create_vector_embedding():
    faiss_store_path = "faiss_store"

    # Load embeddings lazily (CPU) and store in session state
    if "embeddings_model" not in st.session_state:
        try:
            st.session_state.embeddings_model = HuggingFaceEmbeddings(
                model_name="models/kanoon_embedder",
                model_kwargs={"device": "cpu"}
            )
        except Exception as e:
            st.error(f"Error loading embeddings model: {e}")
            return

    # Load documents
    st.session_state.loader = PyPDFDirectoryLoader("dataset")
    st.session_state.docs = st.session_state.loader.load()
    if not st.session_state.docs:
        st.warning("No PDF documents found in the 'dataset' directory.")
        return
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    # Create or load FAISS store
    if os.path.exists(faiss_store_path):
        st.session_state.vector_store = FAISS.load_local(
            faiss_store_path, st.session_state.embeddings_model, allow_dangerous_deserialization=True
        )
    else:
        st.session_state.vector_store = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings_model
        )
        st.session_state.vector_store.save_local(faiss_store_path)
    st.success("Vector store ready!")

# --- Use vector store in chat ---
if "vector_store" in st.session_state:
    vector_store = st.session_state.vector_store
else:
    vector_store = None


# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message['content'])
    else:
        with st.chat_message("assistant", avatar="⚖️"):
            # Render the content with styled card
            st.markdown(message["content"], unsafe_allow_html=True)
            
            # Render sources if available
            if "sources" in message and message["sources"]:
                st.markdown('<div class="section-header">📚 Referenced Sources</div>', unsafe_allow_html=True)
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">{source['icon']} {source['type']}</div>
                            <div class="source-query">Query: "{source['query']}"</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Render debug info if available
            if "debug" in message and message["debug"]:
                with st.expander("🛠️ View Agent Execution Log"):
                    st.json(message["debug"])
                    if "trace" in message:
                        for idx, msg in enumerate(message["trace"]):
                            st.write(f"**Message {idx + 1}:**", msg)

# Handle Chat Input
if user_prompt := st.chat_input("Ask a legal question..."):
    if not vector_store:
        st.error("⚠️ Please build the Knowledge Base first using the Knowledge Base button.")
    else:
        # Add User Message to History
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # Generate Response
        with st.chat_message("assistant", avatar="⚖️"):
            # Use status container for live updates
            with st.status("🔍 Analyzing legal sources...", expanded=True) as status:
                # Initialize Agent
                agent_executor = create_agent(vector_store)
                
                # Initialize callback handler for live tool logging
                st_callback = StreamlitCallbackHandler(st.container())
                
                # Invoke the Agent
                start = time.process_time()
                
                # Prepare chat history (limit to last 10 messages to avoid token limits)
                chat_history = []
                for msg in st.session_state.messages[-10:]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        # For assistant messages, we only need the content, not the full HTML/metadata
                        # Extract text content if it's HTML (simplified approach)
                        content = msg["content"]
                        if "answer-content" in content:
                            # Try to extract just the text from the HTML if possible, 
                            # or just pass a placeholder to indicate previous answer.
                            # For now, we'll pass a summarized version or the full content if it's not too long.
                            chat_history.append(AIMessage(content="[Previous Legal Analysis provided]"))
                        else:
                            chat_history.append(AIMessage(content=content))
                
                # Add current prompt
                chat_history.append(HumanMessage(content=user_prompt))
                
                initial_state = {"messages": chat_history}
                
                # Pass callback to agent
                final_state = agent_executor.invoke(
                    initial_state,
                    config={"callbacks": [st_callback]}
                )
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                
                final_message = final_state["messages"][-1]
                answer_text = final_message.content
                
                # Extract sources and check for predictions
                sources = []
                has_prediction = False
                
                for msg in final_state["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if tool_call['name'] == 'search_legal_docs':
                                sources.append({
                                    "type": "Local PDF Database",
                                    "query": tool_call['args'].get('query'),
                                    "icon": "📄"
                                })
                            elif tool_call['name'] == 'search_indian_kanoon':
                                sources.append({
                                    "type": "Indian Kanoon API",
                                    "query": tool_call['args'].get('query'),
                                    "icon": "🌐"
                                })
                            elif tool_call['name'] == 'predict_case_outcome':
                                has_prediction = True
                                sources.append({
                                    "type": "Outcome Prediction Model",
                                    "query": tool_call['args'].get('case_description'),
                                    "icon": "🎯"
                                })
                
                duration = time.process_time() - start
            
            # Use Groq to generate a proper summary of the legal analysis
            def create_groq_summary(text):
                """Use Groq LLM to generate an accurate summary that preserves key predictions"""
                if len(text) <= 500:
                    return None  # Don't summarize short texts
                
                try:
                    # Initialize Groq LLM for summarization
                    summarizer_llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="llama-3.3-70b-versatile",  # Updated to current supported model
                        temperature=0.3
                    )
                    
                    # Create enhanced summarization prompt that preserves predictions
                    summary_prompt = f"""You are a senior legal expert. Create a comprehensive and detailed legal summary of the following analysis.
                    
CRITICAL INSTRUCTIONS:
1. **Structure**: You must use the following headers:
   - **Executive Summary**: A high-level overview of the case/situation.
   - **Key Legal Grounds**: The specific laws, sections, and precedents applied.
   - **Prediction Analysis**: Who is likely to win, the estimated probability (if available), and why.
   - **Strategic Implications**: Practical advice and next steps for the parties.

2. **Detail Level**: Do NOT be brief. Provide sufficient detail for a lawyer to understand the core arguments.
3. **Tone**: Professional, authoritative, and objective.
4. **Naming Convention**: NEVER use generic placeholders like "X" or "Y". Use specific legal terms (Plaintiff/Defendant/Appellant/Respondent) or actual names if available.

Legal Analysis:
{text}

Generate the detailed summary now:"""
                    
                    # Generate summary
                    summary_response = summarizer_llm.invoke(summary_prompt)
                    return summary_response.content.strip()
                except Exception as e:
                    st.warning(f"Could not generate summary: {e}")
                    # Fallback to first 400 characters
                    return text[:400] + "..." if len(text) > 400 else None
            
            # Generate summary using Groq
            answer_summary = create_groq_summary(answer_text)
            show_summary = answer_summary is not None
            
            # Create styled answer card HTML with summary
            if show_summary:
                # Link citations in summary and full text
                linked_summary = link_citations(answer_summary)
                linked_full_text = link_citations(answer_text)
                
                answer_html = f"""
                <div class="answer-card">
                    <h3>📝 Legal Analysis <span class="time-badge">⚡ {duration:.2f}s</span></h3>
                    <div class="summary-box">
                        <h4>Executive Summary</h4>
                        <p>{linked_summary}</p>
                    </div>
                    <div class="answer-content">
                        {linked_full_text}
                    </div>
                </div>
                """
            else:
                linked_text = link_citations(answer_text)
                answer_html = f"""
                <div class="answer-card">
                    <h3>📝 Legal Analysis <span class="time-badge">⚡ {duration:.2f}s</span></h3>
                    <div class="answer-content">
                        {linked_text}
                    </div>
                </div>
                """
            
            # Display Answer
            st.markdown(answer_html, unsafe_allow_html=True)
            
            # Add expander for full analysis if summary was shown
            if show_summary:
                with st.expander("📖 View Full Detailed Analysis"):
                    st.markdown(f'<div style="line-height: 1.6; color: #1f2937;">{linked_full_text}</div>', unsafe_allow_html=True)
            
            # --- New Features: Download Report ---
            # Download Report
            report_text = f"""LEGAL ANALYSIS REPORT
Generated by AI Research Engine
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}

QUERY:
{user_prompt}

EXECUTIVE SUMMARY:
{answer_summary if show_summary else "N/A"}

FULL ANALYSIS:
{answer_text}

DISCLAIMER: This is an AI-generated report and does not constitute professional legal advice.
"""
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"legal_analysis_{int(time.time())}.txt",
                mime="text/plain"
            )
            
            # Display Sources
            if sources:
                st.markdown('<div class="section-header">📚 Referenced Sources</div>', unsafe_allow_html=True)
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">{source['icon']} {source['type']}</div>
                            <div class="source-query">Query: "{source['query']}"</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Debug Info
            debug_info = {
                "total_messages": len(final_state["messages"]),
                "sources_used": len(sources),
                "execution_time": f"{duration:.3f}s"
            }
            
            with st.expander("🛠️ View Agent Execution Log"):
                st.json(debug_info)
                for idx, msg in enumerate(final_state["messages"]):
                    st.write(f"**Message {idx + 1}:**", msg)
            
            # Save to History
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_html,
                "sources": sources,
                "debug": debug_info,
                "trace": [str(m) for m in final_state["messages"]]
            })

