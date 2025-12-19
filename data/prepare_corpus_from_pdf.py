import os, json, pathlib
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

# Directory containing PDF files
pdf_dir = "dataset"
loader = PyPDFDirectoryLoader(pdf_dir)
raw_docs = loader.load()

# Basic cleaning – strip excessive whitespace
for doc in raw_docs:
    doc.page_content = doc.page_content.strip()

# Split into chunks suitable for retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(raw_docs)

# Optional: save cleaned chunks for reproducibility
out_dir = pathlib.Path("cleaned_chunks_pdf")
out_dir.mkdir(exist_ok=True)
for i, chunk in enumerate(chunks):
    (out_dir/f"{i}.json").write_text(json.dumps(chunk.dict()), encoding="utf-8")

# Build FAISS index using a base embedding model (will be fine‑tuned later)
base_model = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, base_model)
vector_store.save_local("faiss_store")
print(f"FAISS index created from PDFs with {len(chunks)} chunks")
