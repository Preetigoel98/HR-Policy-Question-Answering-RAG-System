# Step 4: Create embeddings and store in ChromaDB

from PyPDF2 import PdfReader
import os
from sentence_transformers import SentenceTransformer
import chromadb

# Settings
pdf_folder = "data/hr_policies"
CHUNK_SIZE = 500
OVERLAP = 50


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# Step 1: Load and chunk PDFs
print("📂 Loading PDFs...")
all_chunks = []
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)

    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "content": chunk,
            "source": pdf_file,
            "chunk_id": i
        })
    print(f"✅ {pdf_file}: {len(chunks)} chunks")

print(f"\n📊 Total chunks: {len(all_chunks)}\n")

# Step 2: Load embedding model
print("🤖 Loading embedding model (first time will download ~80MB)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!\n")

# Step 3: Generate embeddings
print("🔢 Generating embeddings...")
texts = [chunk["content"] for chunk in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)
print(f"✅ Generated {len(embeddings)} embeddings\n")

# Step 4: Store in ChromaDB
print("💾 Creating ChromaDB database...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="hr_policies")

# Add to database
ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in all_chunks]
metadatas = [{"source": chunk["source"], "chunk_id": chunk["chunk_id"]} for chunk in all_chunks]

collection.add(
    embeddings=embeddings.tolist(),
    documents=texts,
    ids=ids,
    metadatas=metadatas
)

print(f"✅ Stored {len(all_chunks)} chunks in ChromaDB")
print(f"📁 Database saved to: ./chroma_db")
print("\n🎉 Vector database created successfully!")