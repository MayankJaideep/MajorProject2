from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uvicorn
import shutil
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# Import core agent
from core_agent import create_agent
from outcome_predictor import OutcomePredictor

app = FastAPI(title="Legal AI Research Engine API", version="2.1") # Bump version to force reload

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class AppState:
    vector_store = None
    embeddings_model = None

state = AppState()

# Models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = [] # [{'role': 'user', 'content': '...'}, ...]

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]] = []

class PredictionRequest(BaseModel):
    description: str
    court: Optional[str] = None
    judge: Optional[str] = None
    case_type: Optional[str] = None
    model_version: str = "advanced" # 'advanced' or 'legacy'

class PredictionResponse(BaseModel):
    result: Dict[str, Any]

# Startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Legal AI Engine...")
    try:
        # Load Embeddings
        try:
            state.embeddings_model = HuggingFaceEmbeddings(
                model_name="models/kanoon_embedder",
                model_kwargs={"device": "cpu"}
            )
        except Exception:
            print("⚠️ 'models/kanoon_embedder' not found or invalid. Falling back to 'all-MiniLM-L6-v2'.")
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
        else:
            print("⚠️ Vector Store not found. Upload PDFs to initialize.")
            
    except Exception as e:
        print(f"❌ Error loading resources: {e}")

# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "active", "vector_store": state.vector_store is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Initialize Agent with current vector store (can be None)
        agent_executor = create_agent(state.vector_store)
        
        # Convert history
        langchain_history = []
        for msg in request.history[-10:]: # Limit history
            if msg['role'] == 'user':
                langchain_history.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_history.append(AIMessage(content=msg['content']))
        
        # Add current message
        langchain_history.append(HumanMessage(content=request.message))
        
        # Invoke Agent
        initial_state = {"messages": langchain_history}
        final_state = agent_executor.invoke(initial_state)
        
        # Extract response
        final_message = final_state["messages"][-1].content
        
        # Extract sources from tool calls
        sources = []
        for msg in final_state["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    sources.append({
                        "type": tool_call['name'],
                        "args": str(tool_call['args'])
                    })
        
        return ChatResponse(response=final_message, sources=sources)
        
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
        # Merge manual overrides
        extracted_features.update({k:v for k,v in features.items() if v})
        
        predictor = OutcomePredictor(model_dir="models")
        result = predictor.predict(extracted_features, model_version=request.model_version)
        
        return {"result": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    try:
        from langchain_community.document_loaders import PyPDFLoader
        
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        documents = []
        
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Load and split
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)
        
        # Update Vector Store
        if state.vector_store:
            state.vector_store.add_documents(final_documents)
        else:
            state.vector_store = FAISS.from_documents(final_documents, state.embeddings_model)
            
        # Save
        state.vector_store.save_local("faiss_store")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return {"message": f"Processed {len(files)} files successfully.", "chunks": len(final_documents)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
