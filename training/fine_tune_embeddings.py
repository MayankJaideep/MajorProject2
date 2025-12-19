import os, json, pathlib
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

# Load PDFs and split into chunks (same as preparation script)
pdf_dir = "dataset"
loader = PyPDFDirectoryLoader(pdf_dir)
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(raw_docs)

# Create training examples: use the title (if any) as query and the chunk text as target
examples = []
for chunk in chunks:
    title = chunk.metadata.get("title", "")
    if title:
        examples.append(InputExample(texts=[title, chunk.page_content]))
    else:
        # fallback: use first 30 chars as pseudo‑query
        pseudo_query = chunk.page_content[:30]
        examples.append(InputExample(texts=[pseudo_query, chunk.page_content]))

print(f"Training on {len(examples)} examples")

model = SentenceTransformer("all-MiniLM-L6-v2")
train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
train_loss = MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    output_path="models/kanoon_embedder",
)
print("Fine‑tuned embedding model saved to models/kanoon_embedder")
