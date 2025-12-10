# Step 5: Search the vector database

from sentence_transformers import SentenceTransformer
import chromadb

# Load model and database
print("🤖 Loading model and database...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="hr_policies")
print("✅ Ready!\n")

# Test queries
test_questions = [
    "What is the work from home policy?",
    "How do I get reimbursement for expenses?",
    "What are the leave policies?"
]

print("=" * 60)
print("🔍 TESTING SEARCH")
print("=" * 60)

for question in test_questions:
    print(f"\n❓ Question: {question}")
    print("-" * 60)

    # Convert question to embedding
    query_embedding = model.encode([question])[0]

    # Search database
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3  # Get top 3 matches
    )

    # Display results
    for i in range(len(results['documents'][0])):
        source = results['metadatas'][0][i]['source']
        content = results['documents'][0][i]
        distance = results['distances'][0][i]

        print(f"\n📄 Result {i + 1} - Source: {source}")
        print(f"   Relevance Score: {1 - distance:.3f}")
        print(f"   Content: {content[:200]}...")

print("\n" + "=" * 60)
print("🎉 Search test complete!")