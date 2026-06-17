import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

with open("persona.json") as f:
    persona = f.read()

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "conversation_topics"
)

chunk_collection = client.get_or_create_collection(
    "message_chunks"
)

def ask_question(query):

    query_embedding = embed_model.encode(
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

    topic_context = "\n\n".join(
        topic_results["documents"][0]
   )

    chunk_context = "\n\n".join(
        chunk_results["documents"][0]
   )

    prompt = f"""
    You are a conversation analysis assistant.

    Answer ONLY from the retrieved evidence.

    Do not invent facts.

    Persona:
    {persona}

    Topic Summaries:
    {topic_context}

    Message Chunks:
    {chunk_context}

    Question:
    {query}

    Provide:

    1. Answer :

    2. Supporting Evidence :
       - bullet 1
       - bullet 2

    3. Confidence :
       - High
       - Medium
       - Low
   """
    response = model.generate_content(prompt)

    return response.text