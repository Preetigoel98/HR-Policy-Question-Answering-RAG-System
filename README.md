# HR-Policy-Question-Answering-RAG-System
A Retrieval-Augmented Generation (RAG) project built using Python, ChromaDB, LangChain, and FastAPI.

📌 Project Description

This project is an HR Policy Question-Answering System that allows users to ask natural-language questions like:

“How many casual leave days are allowed?”

“What is the work-from-home policy?”

“How do I claim expense reimbursement?”

The system searches through HR documents (PDFs) and responds with accurate answers, referencing the actual policy text.

This is done using a Retrieval-Augmented Generation (RAG) pipeline:

HR PDFs → Extract Text

Split Text into Chunks

Convert Chunks to Embeddings

Store Embeddings in ChromaDB vector database

User asks a question

System retrieves the most relevant chunks

Generates a final answer using OpenAI LLM + retrieved context

This improves accuracy and ensures answers always come from the real HR documents.

🚀 Features

✔ Extracts text from HR PDFs using PyMuPDF
✔ Cleans & preprocesses document text
✔ Uses all-MPNet-base-v2 embeddings
✔ Stores and queries vectors in ChromaDB
✔ Fast & accurate retrieval of HR policy content
✔ FastAPI backend for Q&A
✔ Supports multiple HR documents:

Employee Handbook

WFH Policy

Expense Reimbursement Policy

HR Policy Manual (208 pages)

📂 Project Structure
hr_rag_project/
│
├── data/
│   ├── docs/              # Original PDFs
│   ├── docs_txt/          # Extracted text files
│   └── chroma_db/         # Vector database storage
│
├── src/
│   ├── extract_text.py    # PDF → text
│   ├── embeddings_store.py# Create embeddings + store in ChromaDB
│   ├── rag_pipeline.py    # Retrieval + generation logic
│   └── api.py             # FastAPI app for Q&A
│
├── venv/                  # Virtual environment (ignored in git)
├── README.md              # Project documentation
└── requirements.txt       # Dependencies

🛠️ Tech Stack
Component	Tool
Language	Python 3.10+
Vector DB	ChromaDB
Embeddings	Sentence Transformers (all-mpnet-base-v2)
LLM	OpenAI GPT-4.1 / GPT-3.5
Backend API	FastAPI
PDF reader	PyMuPDF

📊 Future Improvements

Multi-language support

Policy update monitoring

Role-based access system

Admin dashboard.
