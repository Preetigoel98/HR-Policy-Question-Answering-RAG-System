# Interactive RAG with LLM - Better version

from sentence_transformers import SentenceTransformer
import chromadb
import subprocess
import re

# Load model and database
print("🤖 Loading HR Assistant with AI...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="hr_policies")
print("✅ Ready!\n")


def clean_text(text):
    """Clean text to remove problematic characters"""
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '')
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    return text.strip()


def search_and_answer(question, top_k=3):
    """Search and generate AI answer"""
    # Step 1: Search
    query_embedding = model.encode([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Step 2: Build context (cleaned)
    context_parts = []
    for i in range(len(results['documents'][0])):
        source = results['metadatas'][0][i]['source']
        content = clean_text(results['documents'][0][i][:500])  # Limit and clean
        context_parts.append(f"Source: {source}\n{content}")

    context = "\n\n".join(context_parts)

    # Step 3: Create simple prompt
    prompt = f"""Based on these HR policy documents, answer the question concisely.

Documents:
{context}

Question: {question}

Answer in 2-3 sentences:"""

    # Step 4: Call Ollama
    print("\n🤖 Generating AI answer...")
    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.2:3b', prompt],
            capture_output=True,
            text=True,
            timeout=60  # Increased timeout
        )

        answer = result.stdout.strip()

        # Display results
        print("\n" + "=" * 70)
        print(f"❓ Question: {question}")
        print("=" * 70)
        print(f"\n💬 AI Answer:\n{answer}\n")
        print("📚 Sources:")
        for i in range(len(results['documents'][0])):
            source = results['metadatas'][0][i]['source']
            print(f"   • {source}")
        print("=" * 70 + "\n")

    except subprocess.TimeoutExpired:
        print("⚠️ Timeout - model took too long. Try a shorter question.\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")


# Interactive loop
print("🤖 HR ASSISTANT WITH AI")
print("=" * 70)
print("Ask questions about HR policies - AI will generate answers!")
print("Type 'quit' to exit.\n")

while True:
    question = input("💬 Ask: ").strip()

    if question.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break

    if not question:
        continue

    search_and_answer(question)