import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

# Load PDFs and split into chunks
pdf_dir = "dataset"
loader = PyPDFDirectoryLoader(pdf_dir)
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(raw_docs)

print(f"Loaded {len(chunks)} chunks from PDFs")

# Load the fine-tuned embedding model using HuggingFaceEmbeddings
print("Loading fine-tuned embedding model from models/kanoon_embedder...")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="models/kanoon_embedder",
    model_kwargs={"device": "cpu"}
)

# Create FAISS index with fine-tuned embeddings
print("Building FAISS index with fine-tuned embeddings...")
vector_store = FAISS.from_documents(chunks, embeddings)

# Save the new index
vector_store.save_local("faiss_store")
print("✅ FAISS index rebuilt with fine-tuned embeddings and saved to faiss_store/")
