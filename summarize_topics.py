import json

with open("topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

for topic in topics[:10]:

    text = topic["text"][:3000]

    print("="*80)
    print(text[:500])