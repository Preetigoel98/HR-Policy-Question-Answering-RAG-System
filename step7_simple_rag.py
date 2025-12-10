# Step 7: Simple RAG without LLM (shows retrieved chunks)

from sentence_transformers import SentenceTransformer
import chromadb

# Load model and database
print("🤖 Loading...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="hr_policies")
print("✅ Ready!\n")


def ask_hr_question(question, top_k=3):
    """Search and display relevant policy sections"""
    # Search
    query_embedding = model.encode([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    print(f"❓ Question: {question}")
    print("=" * 70)

    for i in range(len(results['documents'][0])):
        source = results['metadatas'][0][i]['source']
        content = results['documents'][0][i]
        score = 1 - results['distances'][0][i]

        print(f"\n📄 Result {i + 1} | Source: {source} | Score: {score:.2f}")
        print("-" * 70)
        print(content[:400] + "...")

    print("\n" + "=" * 70 + "\n")


# Test questions
questions = [
    "What is the work from home policy?",
    "How do I claim travel expenses?",
    "What is the leave policy?",
    "Can I work remotely?"
]

print("🤖 HR POLICY ASSISTANT")
print("=" * 70 + "\n")

for q in questions:
    ask_hr_question(q)

print("✅ Done!")