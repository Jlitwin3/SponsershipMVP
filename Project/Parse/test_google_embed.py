import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("No GOOGLE_API_KEY found")
    exit()

client = genai.Client(api_key=api_key)

try:
    result = client.models.embed_content(
        model="text-embedding-004",
        contents="test query"
    )
    print("Google Embedding Success!")
    print(f"Dimension: {len(result.embeddings[0].values)}")
except Exception as e:
    print(f"Google Embedding Failed: {e}")
