import json
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

with open("topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

sample_text = ""

for topic in topics[:100]:
    sample_text += topic["text"][:1000] + "\n"

prompt = f"""
Analyze the conversation.

Extract ONLY information supported by the text.

Return valid JSON:

{{
  "habits": [],
  "personal_facts": [],
  "personality_traits": [],
  "communication_style": {{
      "message_length": "",
      "tone": "",
      "emoji_usage": ""
  }}
}}

Conversation:
{sample_text[:20000]}
"""

response = model.generate_content(prompt)

print(response.text)