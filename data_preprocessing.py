import pandas as pd
import numpy as np

df = pd.read_csv("conversations.csv", header=None)

all_messages = []

msg_id = 0

for row_id, row in df.iterrows():

    conversation = str(row[0])

    msgs = conversation.split("\n")

    for msg in msgs:

        all_messages.append({
            "message_id": msg_id,
            "row_id": row_id,
            "text": msg.strip()
        })

        msg_id += 1

print("Total Messages:", len(all_messages))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [m["text"] for m in all_messages]

WINDOW = 5

blocks = []

for i in range(0, len(texts), WINDOW):

    block_text = " ".join(
        texts[i:i+WINDOW]
    )

    blocks.append(block_text) 

block_embeddings = model.encode(
    blocks,
    batch_size=64,
    show_progress_bar=True
)

np.save(
    "block_embeddings.npy",
    block_embeddings
)

print("Saved block embeddings")
print(block_embeddings.shape)

import json

checkpoints = []

for i in range(0, len(texts), 100):

    checkpoints.append({
        "start_message": i,
        "end_message": min(i+99, len(texts)-1),
        "text": " ".join(texts[i:i+100])
    })

with open(
    "message_checkpoints.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(checkpoints, f, indent=2)

print("Checkpoints:", len(checkpoints))

import json
chunks = []

for i in range(0, len(all_messages), 20):

    chunk_text = "\n".join(
        [m["text"] for m in all_messages[i:i+20]]
    )

    chunks.append({
        "chunk_id": i // 20,
        "text": chunk_text
    })

with open(
    "message_chunks.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(chunks, f, indent=2)

chunk_texts = [c["text"] for c in chunks]

chunk_embeddings = model.encode(
    chunk_texts,
    batch_size=64,
    show_progress_bar=True
)

np.save(
    "chunk_embeddings.npy",
    chunk_embeddings
)