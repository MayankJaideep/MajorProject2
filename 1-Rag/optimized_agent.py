#!/usr/bin/env python3

import os
import time
import streamlit as st
from typing import Annotated, List, Literal, TypedDict, Union
from functools import lru_cache

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

# Import performance monitor
from performance_optimizer import perf_monitor, cached_api_call

# Import enhanced components
try:
    from enhanced_feature_extractor import EnhancedFeatureExtractor
    feature_extractor = EnhancedFeatureExtractor()
    print("✅ Using Enhanced Feature Extractor (20 features)")
except ImportError:
    from feature_extractor import FeatureExtractor
    feature_extractor = FeatureExtractor()
    print("⚠️  Using Basic Feature Extractor (6 features)")

from langgraph.graph.message import add_messages
from outcome_predictor import OutcomePredictor
from legal_summarizer import LegalSummarizer
from legal_ner import LegalNER
from historical_analyzer import HistoricalAnalyzer

# Define the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def create_agent(vector_store):
    """
    Creates an optimized LangGraph agent with better performance and accuracy.
    """
    
    # 1. Define optimized tools with caching
    @tool
    def search_legal_docs(query: str):
        """
        Searches the internal legal database for relevant case laws, statutes, and documents.
        Use this tool to find information to answer the user's legal questions.
        """
        perf_monitor.start_timer()
        print(f"DEBUG: Tool called with query: {query}")
        
        try:
            # Use optimized search with top_k=5 for faster results
            docs = vector_store.similarity_search(query, k=5)
            
            if not docs:
                perf_monitor.end_timer("vector_searches")
                return "No relevant documents found in the local database."
            
            # Format results more efficiently
            results = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content[:500]  # Limit content for faster processing
                results.append(f"Source: {source}\nContent: {content}...")
            
            content = "\n\n".join(results)
            perf_monitor.end_timer("vector_searches")
            perf_monitor.metrics["vector_searches"] += 1
            return content
            
        except Exception as e:
            perf_monitor.end_timer("vector_searches")
            return f"Error searching documents: {str(e)}"

    @tool
    @lru_cache(maxsize=128)  # Cache Indian Kanoon results
    def search_indian_kanoon(query: str):
        """
        Searches the Indian Kanoon database for legal cases and statutes.
        Use this tool when you need external legal information or recent judgments not found in the local database.
        """
        perf_monitor.start_timer()
        print(f"DEBUG: Indian Kanoon Tool called with query: {query}")
        
        api_token = os.getenv("INDIAN_KANOON_API_TOKEN")
        if not api_token:
            return "Error: Indian Kanoon API token not found."
        
        # Use cached API call
        def make_api_call():
            import requests
            url = "https://api.indiankanoon.org/search/"
            headers = {"Authorization": f"Token {api_token}"}
            params = {"formInput": query, "pagenum": 0}
            
            response = requests.post(url, headers=headers, data=params, timeout=5)  # Reduced timeout
            response.raise_for_status()
            return response.json()
        
        try:
            data = cached_api_call(make_api_call, f"indian_kanoon_{query}")
            
            # Parse results efficiently
            docs = data.get('docs', [])[:3]  # Reduced from 5 to 3 for speed
            if not docs:
                perf_monitor.end_timer("api_calls")
                return "No relevant documents found on Indian Kanoon."
            
            results = []
            for doc in docs:
                title = doc.get('title', 'No Title')[:100]  # Limit title length
                snippet = doc.get('headline', 'No Snippet')[:200]  # Limit snippet length
                results.append(f"Title: {title}\nSnippet: {snippet}\n")
            
            content = "\n\n".join(results)
            perf_monitor.end_timer("api_calls")
            perf_monitor.metrics["api_calls"] += 1
            return content
            
        except Exception as e:
            perf_monitor.end_timer("api_calls")
            return f"Error searching Indian Kanoon: {str(e)}"

    @tool
    def predict_case_outcome(case_description: str):
        """
        Predicts the outcome of a legal case using Random Forest with 20 features.
        Use this tool when the user asks about chances of winning, outcome prediction, or success probability.
        
        Args:
            case_description: Description of the case including court, case type, and other relevant details
        
        Returns:
            Prediction with probabilities and confidence score
        """
        perf_monitor.start_timer()
        print(f"DEBUG: Outcome Prediction Tool called with: {case_description}")
        
        try:
            # Use cached feature extraction
            cache_key = f"features_{hash(case_description)}"
            if hasattr(st.session_state, 'feature_cache') and cache_key in st.session_state.feature_cache:
                features = st.session_state.feature_cache[cache_key]
            else:
                features = feature_extractor.extract_all_features(case_description)
                if 'feature_cache' not in st.session_state:
                    st.session_state.feature_cache = {}
                st.session_state.feature_cache[cache_key] = features
            
            # Load and use predictor with error handling
            predictor = OutcomePredictor(model_dir="models")
            predictor.load_model()
            
            # Make prediction
            result = predictor.predict(features)
            
            perf_monitor.end_timer("ml_predictions")
            perf_monitor.metrics["ml_predictions"] += 1
            
            return result
            
        except Exception as e:
            perf_monitor.end_timer("ml_predictions")
            return f"Error predicting outcome: {str(e)}"

    @tool
    def summarize_document(text: str):
        """
        Summarizes legal documents using advanced NLP techniques.
        Use this tool when users need concise summaries of long legal texts.
        """
        perf_monitor.start_timer()
        try:
            summarizer = LegalSummarizer()
            result = summarizer.summarize(text)
            perf_monitor.end_timer("ml_predictions")
            perf_monitor.metrics["ml_predictions"] += 1
            return result
        except Exception as e:
            perf_monitor.end_timer("ml_predictions")
            return f"Error summarizing document: {str(e)}"

    @tool
    def extract_entities(text: str):
        """
        Extracts legal entities (courts, judges, acts, sections) from text.
        Use this tool to identify key legal entities in documents.
        """
        perf_monitor.start_timer()
        try:
            ner = LegalNER()
            entities = ner.extract_entities(text)
            perf_monitor.end_timer("ml_predictions")
            perf_monitor.metrics["ml_predictions"] += 1
            
            # Format entities as clean text instead of raw dict
            if 'error' in entities:
                return entities['error']
            
            formatted_entities = []
            for entity_type, entity_list in entities.items():
                if entity_list:  # Only show non-empty entity types
                    entity_names = [e['text'] for e in entity_list[:3]]  # Limit to top 3
                    formatted_entities.append(f"{entity_type}: {', '.join(entity_names)}")
            
            if formatted_entities:
                return "Key entities identified: " + "; ".join(formatted_entities)
            else:
                return "No specific legal entities identified in the text."
                
        except Exception as e:
            perf_monitor.end_timer("ml_predictions")
            return f"Error extracting entities: {str(e)}"

    # 2. Create tool list
    tools = [search_legal_docs, search_indian_kanoon, predict_case_outcome, summarize_document, extract_entities]
    
    # 3. Create optimized LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Use faster model
        temperature=0.1,  # Lower temperature for more consistent responses
        max_tokens=1024  # Limit tokens for faster responses
    )
    
    # 4. Create tool node
    tool_node = ToolNode(tools)
    
    # 5. Define optimized agent logic
    def should_continue(state: AgentState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    def call_model(state: AgentState):
        messages = state["messages"]
        
        # Enhanced system prompt for better accuracy and cleaner responses
        system_prompt = """You are an expert Legal AI Assistant specializing in Indian law. Your role is to provide accurate, well-structured legal information.

RESPONSE GUIDELINES:
1. Provide clear, professional legal analysis
2. For case analysis, ALWAYS include a "Winning Prediction" section
3. Cite relevant laws and case precedents when possible
4. Structure responses with headings and bullet points
5. Include practical legal implications
6. NEVER show debug information or raw tool outputs
7. Format entity extraction results naturally within the response
8. NEVER use generic placeholders like 'X' or 'Y'. Use specific legal terms (Plaintiff/Defendant/Appellant/Respondent) or actual names if available.

WINNING PREDICTION REQUIREMENTS:
- For any case analysis, predict who is likely to win
- Provide confidence level (High/Medium/Low)
- List key factors supporting the prediction
- Consider legal precedents and applicable laws
- Be specific about the reasoning

AVAILABLE TOOLS:
- Local legal database search (for Indian statutes)
- Indian Kanoon API (for recent case law)
- ML outcome prediction (for case analysis)
- Document summarization (for complex legal texts)
- Entity extraction (for identifying key legal elements)

Use the most relevant tools to provide comprehensive, accurate legal information. Focus on practical legal insights rather than technical details."""
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    # 6. Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app
