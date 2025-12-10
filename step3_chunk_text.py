# Step 3: Chunk text into smaller pieces

from PyPDF2 import PdfReader
import os

# Settings
pdf_folder = "data/hr_policies"
CHUNK_SIZE = 500  # words per chunk
OVERLAP = 50  # overlapping words between chunks


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# Load and chunk all PDFs
all_chunks = []
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

print(f"📂 Processing {len(pdf_files)} PDFs...\n")

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    reader = PdfReader(pdf_path)

    # Extract all text
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    # Chunk the text
    chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)

    # Store chunks with metadata
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "content": chunk,
            "source": pdf_file,
            "chunk_id": i
        })

    print(f"✅ {pdf_file}")
    print(f"   Total words: {len(full_text.split())}")
    print(f"   Created {len(chunks)} chunks")
    print(f"   Sample chunk: {chunks[0][:100]}...\n")

print(f"\n🎉 Total chunks created: {len(all_chunks)}")
print(f"📊 Average chunk size: {sum(len(c['content'].split()) for c in all_chunks) // len(all_chunks)} words")