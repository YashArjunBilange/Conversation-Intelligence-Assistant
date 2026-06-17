import json
import time
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

with open("topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

topic_summaries = []

for idx, topic in enumerate(topics):

    try:

        prompt = f"""
        Summarize this conversation topic.

        Return JSON:
        {{
            "title": "...",
            "summary": "..."
        }}

        Conversation:
        {topic['text'][:4000]}
        """

        response = model.generate_content(prompt)

        topic_summaries.append({
            "topic_id": topic["topic_id"],
            "start_message": topic["start_message"],
            "end_message": topic["end_message"],
            "summary": response.text
        })

        if idx % 20 == 0:
            with open(
                "topic_summaries.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    topic_summaries,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            print(f"Saved {idx} topics")

    except Exception as e:
        print("Error:", e)

    time.sleep(1)