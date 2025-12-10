# Step 2: Load ALL 4 PDFs

from PyPDF2 import PdfReader
import os

# Folder containing PDFs
pdf_folder = "data/hr_policies"

# Get all PDF files
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

print(f"📂 Found {len(pdf_files)} PDF files:\n")

# Load each PDF
all_documents = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    reader = PdfReader(pdf_path)

    # Extract all text from all pages
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    # Store document info
    all_documents.append({
        "filename": pdf_file,
        "num_pages": len(reader.pages),
        "text": full_text
    })

    print(f"✅ {pdf_file}")
    print(f"   Pages: {len(reader.pages)}")
    print(f"   Characters: {len(full_text)}")
    print(f"   Preview: {full_text[:100]}...\n")

print(f"\n🎉 Successfully loaded {len(all_documents)} documents!")