import chromadb
import numpy as np
import json

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="conversation_topics"
)

with open("topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

topic_embeddings = np.load("topic_embeddings.npy")

for i, topic in enumerate(topics):

    collection.add(
        ids=[str(topic["topic_id"])],

        embeddings=[
            topic_embeddings[i].tolist()
        ],

        documents=[
            topic["text"][:4000]
        ],

        metadatas=[
            {
                "start": topic["start_message"],
                "end": topic["end_message"]
            }
        ]
    )

print(collection.count())

with open("message_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load("chunk_embeddings.npy")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    "message_chunks"
)

for i, chunk in enumerate(chunks):

    collection.add(
        ids=[str(chunk["chunk_id"])],
        embeddings=[embeddings[i].tolist()],
        documents=[chunk["text"]]
    )

print("Chunk count:", collection.count())