# Step 1: Load ONE PDF and see what's inside

from PyPDF2 import PdfReader

# Path to one PDF
pdf_path = "data/hr_policies/wfh_policy_sample.pdf"

# Load the PDF
reader = PdfReader(pdf_path)

# Get number of pages
num_pages = len(reader.pages)
print(f"📄 PDF has {num_pages} pages")

# Extract text from first page
first_page = reader.pages[0]
text = first_page.extract_text()

# Show first 500 characters
print("\n📝 First 500 characters from page 1:")
print(text[:500])