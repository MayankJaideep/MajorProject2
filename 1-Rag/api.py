from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uvicorn
import shutil
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from rank_bm25 import BM25Okapi

# Import core agent
from core_agent import create_agent
from outcome_predictor import OutcomePredictor

app = FastAPI(title="Legal AI Research Engine API", version="2.2")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class AppState:
    vector_store = None
    embeddings_model = None
    bm25_index = None
    bm25_corpus = None
    semantic_cache = {} # {query_embedding_bytes: response_object}
    cache_keys = [] # List of embeddings (np.array)
    cache_values = [] # List of response objects

state = AppState()

# Models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]] = []
    cached: bool = False

class PredictionRequest(BaseModel):
    description: str
    court: Optional[str] = None
    judge: Optional[str] = None
    case_type: Optional[str] = None
    model_version: str = "advanced"

class PredictionResponse(BaseModel):
    result: Dict[str, Any]

def update_bm25_index():
    """Helper to rebuild BM25 index from Vector Store documents"""
    if state.vector_store and hasattr(state.vector_store, 'docstore'):
        try:
            print("🔄 Building BM25 Index...")
            # Extract all docs
            docs = list(state.vector_store.docstore._dict.values())
            if not docs:
                return
            
            # Tokenize
            tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
            
            # Build Index
            state.bm25_index = BM25Okapi(tokenized_corpus)
            state.bm25_corpus = docs # Store actual doc objects for retrieval
            print(f"✅ BM25 Index Built ({len(docs)} documents)")
        except Exception as e:
            print(f"⚠️ Failed to build BM25 Index: {e}")

# Startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Legal AI Engine...")
    try:
        # Load Embeddings
        try:
            # Check if InLegalBERT feature extractor initialized it
            # We use langchain wrapper here for vector store compatibility
            state.embeddings_model = HuggingFaceEmbeddings(
                model_name="law-ai/InLegalBERT", # Match the extractor
                model_kwargs={"device": "cpu"}
            )
            print("✅ Embeddings Model Loaded (InLegalBERT)")
        except Exception:
            print("⚠️ InLegalBERT loading failed. Falling back to all-MiniLM-L6-v2.")
            state.embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
        
        # Load Vector Store
        if os.path.exists("faiss_store"):
            state.vector_store = FAISS.load_local(
                "faiss_store", state.embeddings_model, allow_dangerous_deserialization=True
            )
            print("✅ Vector Store Loaded")
            
            # Build BM25
            update_bm25_index()
            
        else:
            print("⚠️ Vector Store not found. Upload PDFs to initialize.")
            
    except Exception as e:
        print(f"❌ Error loading resources: {e}")

# Endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "active", 
        "vector_store": state.vector_store is not None,
        "bm25_index": state.bm25_index is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # --- SEMANTIC CACHE LOOKUP ---
        query_vec = np.array(state.embeddings_model.embed_query(request.message))
        
        # Simple linear scan (fast for <1000 items)
        best_sim = -1.0
        best_idx = -1
        
        for i, cached_vec in enumerate(state.cache_keys):
            # Cosine similarity (A . B) / (|A|*|B|)
            # Assuming vectors are not normalized, but HF embeddings usually are? 
            # Safest is manual norm.
            norm_q = np.linalg.norm(query_vec)
            norm_c = np.linalg.norm(cached_vec)
            if norm_q > 0 and norm_c > 0:
                sim = np.dot(query_vec, cached_vec) / (norm_q * norm_c)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
        
        if best_sim > 0.90:
            print(f"⚡ Cache Hit (Similarity: {best_sim:.4f})")
            return state.cache_values[best_idx]
            
        # --- END CACHE ---

        # Initialize Agent
        agent_executor = create_agent(
            state.vector_store,
            bm25_index=state.bm25_index,
            bm25_corpus=state.bm25_corpus
        )
        
        # Convert history
        langchain_history = []
        for msg in request.history[-10:]:
            if msg['role'] == 'user':
                langchain_history.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_history.append(AIMessage(content=msg['content']))
        
        langchain_history.append(HumanMessage(content=request.message))
        
        initial_state = {"messages": langchain_history}
        final_state = agent_executor.invoke(initial_state)
        
        final_message = final_state["messages"][-1].content
        
        sources = []
        for msg in final_state["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    sources.append({
                        "type": tool_call['name'],
                        "args": str(tool_call['args'])
                    })
        
        response_obj = ChatResponse(response=final_message, sources=sources, cached=False)
        
        # --- UPDATE CACHE ---
        # Evict if full (FIFO)
        if len(state.cache_keys) > 100:
            state.cache_keys.pop(0)
            state.cache_values.pop(0)
            
        state.cache_keys.append(query_vec)
        # Store a copy of response with cached=True for next time
        cached_response = ChatResponse(response=final_message, sources=sources, cached=True)
        state.cache_values.append(cached_response)
        
        return response_obj
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    try:
        features = {
            "description": request.description,
            "court": request.court or "",
            "judge": request.judge or "",
            "case_type": request.case_type or ""
        }
        
        from enhanced_feature_extractor import EnhancedFeatureExtractor
        extractor = EnhancedFeatureExtractor()
        extracted_features = extractor.extract_all_features(request.description)
        extracted_features.update({k:v for k,v in features.items() if v})
        
        predictor = OutcomePredictor(model_dir="models")
        result = predictor.predict(extracted_features, model_version=request.model_version)
        
        return {"result": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        documents = []
        
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)
        
        if state.vector_store:
            state.vector_store.add_documents(final_documents)
        else:
            state.vector_store = FAISS.from_documents(final_documents, state.embeddings_model)
            
        state.vector_store.save_local("faiss_store")
        
        # Rebuild BM25 after upload
        update_bm25_index()
        
        shutil.rmtree(temp_dir)
        
        return {"message": f"Processed {len(files)} files successfully.", "chunks": len(final_documents)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize/timeline")
async def visualize_timeline(request: dict):
    """
    Extract chronological timeline from legal case text.
    
    Input: {"text": "..."}
    Output: {"events": [...], "message": "..."}
    """
    try:
        case_text = request.get("text", "")
        
        if not case_text or len(case_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Please provide sufficient case text (minimum 50 characters)")
        
        from timeline_extractor import TimelineExtractor
        
        extractor = TimelineExtractor()
        timeline_events = extractor.extract_chronology(case_text)
        
        if not timeline_events:
            return {
                "events": [],
                "message": "No dated events found in the provided text. Please ensure the text contains specific dates or time references."
            }
        
        return {
            "events": timeline_events,
            "message": f"Successfully extracted {len(timeline_events)} timeline events"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Timeline extraction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
