import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    "conversation_topics"
)

chunk_collection = client.get_or_create_collection(
    "message_chunks"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

query = "What hobbies does this user have?"

query_embedding = model.encode(
    [query]
).tolist()

topic_results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

chunk_results = chunk_collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

print("\nRetrieved Documents:\n")

for i, doc in enumerate(topic_results["documents"][0]):

    print("=" * 80)
    print(f"Document {i+1}")
    print(doc[:1000])
    print()

print("\n")
print("="*100)
print("CHUNK RESULTS")
print("="*100)

for i, doc in enumerate(chunk_results["documents"][0]):

    print("\n")
    print("-"*80)

    print(f"Chunk {i+1}")

    print(doc[:1000])