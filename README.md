# Conversation Intelligence Assistant

## Overview

This project builds an end-to-end Conversation Intelligence System that analyzes conversations chronologically, detects topic transitions, extracts user personas, and answers questions using a Retrieval-Augmented Generation (RAG) pipeline.

The system processes raw conversations, creates semantic topic checkpoints, stores them in a vector database, and uses Gemini to generate grounded responses based on retrieved evidence.

---

# Features

## Topic Change Detection

The dataset is processed in chronological order.

### Methodology

1. Split conversations into individual messages.
2. Group messages into blocks of 5 messages.
3. Generate semantic embeddings for each block using `all-MiniLM-L6-v2`.
4. Compute cosine similarity between consecutive blocks.
5. Calculate a similarity threshold using the 10th percentile.
6. Create a topic boundary whenever similarity falls below the threshold.
7. Merge nearby boundaries to avoid over-segmentation.
8. Generate topic segments and save them in `topics.json`.

This ensures topic transitions are detected based on semantic changes rather than fixed message counts.

---

## Topic Summarization

Each detected topic segment is summarized using Gemini.

Each topic checkpoint contains:

* Topic ID
* Start Message Index
* End Message Index
* Topic Content
* Topic Summary

The summaries provide compact semantic representations of long conversation sections.

---

## 100-Message Checkpoints

In addition to topic segmentation, the system creates checkpoints every 100 messages.

Purpose:

* Long-range conversation memory
* Context preservation
* Additional retrieval support

Stored in:

`message_checkpoints.json`

---

## Persona Extraction

The system builds a user persona by analyzing all detected topics.

The persona contains:

### Personal Facts

Information explicitly stated by the user.

Examples:

* Occupation
* Location
* Education
* Family information

### Hobbies and Interests

Repeated activities and preferences.

Examples:

* Reading
* Hiking
* Cooking
* Music

### Personality Traits

Inferred behavioral characteristics.

Examples:

* Friendly
* Curious
* Supportive
* Enthusiastic

### Communication Style

Conversation patterns such as:

* Formal vs Informal
* Verbosity
* Enthusiasm
* Question frequency

The generated persona is stored in:

`persona.json`

---

# Retrieval-Augmented Generation (RAG)

The chatbot uses a two-level retrieval strategy.

## Level 1: Topic Retrieval

Topic checkpoints are stored in ChromaDB.

When a question is asked:

1. Query is converted into an embedding.
2. Vector similarity search retrieves the most relevant topics.
3. Retrieved topics provide high-level context.

## Level 2: Message Retrieval

Message chunks are stored separately.

When a question is asked:

1. Relevant chunks are retrieved using vector search.
2. Retrieved chunks provide detailed supporting evidence.

## Answer Generation

The following are provided to Gemini:

* User Persona
* Retrieved Topics
* Retrieved Message Chunks
* User Question

Gemini generates an answer strictly from retrieved evidence.

This minimizes hallucinations and improves factual grounding.

---

# Architecture

Conversation Dataset
↓
Message Parsing
↓
Block Creation (5 Messages)
↓
Sentence Embeddings
↓
Cosine Similarity Analysis
↓
Topic Boundary Detection
↓
Topic Segmentation
↓
Topic Summarization
↓
Persona Extraction
↓
ChromaDB Indexing
↓
Semantic Retrieval
↓
Gemini
↓
Chatbot Response

---

# Technology Stack

* Python
* Pandas
* NumPy
* Sentence Transformers
* ChromaDB
* Scikit-Learn
* Gemini 2.5 Flash
* Streamlit

---

# Project Structure

app.py                     # Streamlit UI

rag_chatbot.py             # Main RAG pipeline

topic_detection.py         # Topic segmentation

summarize_topics.py        # Topic summarization

persona.py                 # Persona generation

chroma_setup.py            # ChromaDB indexing

retriever.py               # Retrieval testing

topics.json                # Topic checkpoints

persona.json               # Persona output

message_chunks.json        # Retrieval chunks

message_checkpoints.json   # 100-message checkpoints

README.md

requirements.txt

---

# Installation

## Clone Repository

git clone <repository-url>

cd <repository-folder>

## Install Dependencies

pip install -r requirements.txt

## Create Environment File

Create a `.env` file:

GEMINI_API_KEY=YOUR_API_KEY

---

# Running the Application

Start Streamlit:

streamlit run app.py

Application will be available at:

http://localhost:8501

---

## Streamlit Cloud Deployment

This project is ready to deploy on Streamlit Community Cloud with `app.py` as the entry point.

Before deploying, add `GEMINI_API_KEY` in Streamlit secrets or the app environment.

The chatbot no longer depends on committed ChromaDB binaries or `.npy` embedding files. If the vector store is empty, it will rebuild it from the tracked JSON sources on first run.

---

# Hosted Demo

Streamlit URL:

<ADD_YOUR_STREAMLIT_URL>

---

# Video Demonstration

Loom Video:

<ADD_YOUR_LOOM_LINK>

---

# Example Questions

* What hobbies does this user have?
* What personality traits are evident?
* How does this user communicate?
* What jobs have they mentioned?
* What locations are important to them?

---

# Evaluation Criteria Addressed

✅ Chronological Topic Splitting

✅ Semantic Topic Detection

✅ Persona Construction

✅ Relevant Vector Retrieval

✅ Retrieval-Augmented Generation

✅ End-to-End Chatbot

✅ Streamlit Deployment

---

# Future Improvements

* Hybrid Keyword + Semantic Retrieval
* Reranking Layer
* Source Citations
* Better Topic Segmentation Models
* Multi-User Persona Tracking
* Conversation Timeline Visualization
