# Interactive HR Assistant - Ask any question!

from sentence_transformers import SentenceTransformer
import chromadb

# Load model and database
print("🤖 Loading HR Assistant...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="hr_policies")
print("✅ Ready!\n")


def search_policies(question, top_k=3):
    """Search for relevant policy sections"""
    query_embedding = model.encode([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    print("\n" + "=" * 70)
    print(f"❓ Your Question: {question}")
    print("=" * 70)

    for i in range(len(results['documents'][0])):
        source = results['metadatas'][0][i]['source']
        content = results['documents'][0][i]
        score = 1 - results['distances'][0][i]

        print(f"\n📄 Result {i + 1} | {source} | Relevance: {score:.2f}")
        print("-" * 70)
        print(content[:400])
        if len(content) > 400:
            print("...")

    print("\n" + "=" * 70 + "\n")


# Interactive loop
print("🤖 HR POLICY ASSISTANT")
print("=" * 70)
print("Type your questions about HR policies.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    question = input("💬 Your Question: ").strip()

    if question.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break

    if not question:
        print("Please enter a question.\n")
        continue

    search_policies(question)