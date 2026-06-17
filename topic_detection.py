import pandas as pd

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

texts = [m["text"] for m in all_messages]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

block_embeddings = np.load("block_embeddings.npy")

sims = []

for i in range(1, len(block_embeddings)):
    sim = cosine_similarity(
        [block_embeddings[i-1]],
        [block_embeddings[i]]
    )[0][0]
    sims.append(sim)

print("Min:", min(sims))
print("Max:", max(sims))
print("Mean:", np.mean(sims))
print("Median:", np.median(sims))

for p in [5,10,15,20,25]:
    print(f"P{p}:", np.percentile(sims,p))

threshold = np.percentile(sims, 10)

boundaries = [
    i+1 for i, sim in enumerate(sims)
    if sim < threshold
]

print("Threshold:", threshold)
print("Boundaries:", len(boundaries))

MIN_BLOCKS = 4

filtered_boundaries = []

last_boundary = 0

for boundary in boundaries:

    if boundary - last_boundary >= MIN_BLOCKS:
        filtered_boundaries.append(boundary)
        last_boundary = boundary

print("Filtered boundaries:", len(filtered_boundaries))

WINDOW = 5

topics = []

start_block = 0

for topic_id, boundary in enumerate(filtered_boundaries):

    start_msg = start_block * WINDOW
    end_msg = boundary * WINDOW

    topic_text = " ".join(
        texts[start_msg:end_msg]
    )

    topics.append({
        "topic_id": topic_id,
        "start_message": start_msg,
        "end_message": end_msg,
        "text": topic_text
    })

    start_block = boundary

topics.append({
    "topic_id": len(topics),
    "start_message": start_block * WINDOW,
    "end_message": len(texts)-1,
    "text": " ".join(texts[start_block*WINDOW:])
})

import json

with open(
    "topics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        topics,
        f,
        ensure_ascii=False,
        indent=2
    )