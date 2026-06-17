from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
TOPICS_PATH = BASE_DIR / "topics.json"
CHUNKS_PATH = BASE_DIR / "message_chunks.json"
PERSONA_PATH = BASE_DIR / "persona.json"
CHROMA_PATH = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


@lru_cache(maxsize=1)
def get_persona_text() -> str:
    if PERSONA_PATH.exists():
        return PERSONA_PATH.read_text(encoding="utf-8")
    return "Persona data is not available."


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_generative_model():
    if not GEMINI_API_KEY:
        return None
    return genai.GenerativeModel("gemini-2.5-flash")


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


@lru_cache(maxsize=1)
def load_topics() -> list[dict]:
    return json.loads(TOPICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def _reset_collection(name: str):
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name)


def _encode_texts(texts: list[str]) -> list[list[float]]:
    return get_embedding_model().encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()


def _populate_topics_collection():
    topics = load_topics()
    collection = get_chroma_client().get_or_create_collection(name="conversation_topics")

    if collection.count() == len(topics):
        return collection

    collection = _reset_collection("conversation_topics")

    batch_size = 64
    for start in range(0, len(topics), batch_size):
        batch = topics[start:start + batch_size]
        texts = [item["text"] for item in batch]
        collection.add(
            ids=[str(item["topic_id"]) for item in batch],
            embeddings=_encode_texts(texts),
            documents=texts,
            metadatas=[
                {
                    "start": item["start_message"],
                    "end": item["end_message"],
                }
                for item in batch
            ],
        )

    return collection


def _populate_chunks_collection():
    chunks = load_chunks()
    collection = get_chroma_client().get_or_create_collection(name="message_chunks")

    if collection.count() == len(chunks):
        return collection

    collection = _reset_collection("message_chunks")

    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [item["text"] for item in batch]
        collection.add(
            ids=[str(item["chunk_id"]) for item in batch],
            embeddings=_encode_texts(texts),
            documents=texts,
        )

    return collection


@lru_cache(maxsize=1)
def get_topic_collection():
    return _populate_topics_collection()


@lru_cache(maxsize=1)
def get_chunk_collection():
    return _populate_chunks_collection()


def _join_documents(results: dict, default_message: str) -> str:
    documents = results.get("documents", [])
    if not documents or not documents[0]:
        return default_message
    return "\n\n".join(documents[0])


def ask_question(query: str) -> str:
    query = query.strip()
    if not query:
        return "Please enter a question before asking for an answer."

    if not GEMINI_API_KEY:
        return "Set GEMINI_API_KEY in the Streamlit secrets or environment before using the app."

    try:
        query_embedding = _encode_texts([query])

        topic_results = get_topic_collection().query(
            query_embeddings=query_embedding,
            n_results=3,
        )

        chunk_results = get_chunk_collection().query(
            query_embeddings=query_embedding,
            n_results=5,
        )

        topic_context = _join_documents(
            topic_results,
            "No relevant topic summaries were found.",
        )

        chunk_context = _join_documents(
            chunk_results,
            "No relevant message chunks were found.",
        )

        prompt = f"""
You are a conversation analysis assistant.

Answer only from the retrieved evidence.
Do not invent facts.

Persona:
{get_persona_text()}

Topic Summaries:
{topic_context}

Message Chunks:
{chunk_context}

Question:
{query}

Provide:

1. Answer
2. Supporting Evidence
3. Confidence
""".strip()

        response = get_generative_model().generate_content(prompt)
        return response.text
    except Exception as exc:
        return f"Unable to answer right now: {exc}"