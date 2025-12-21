import os
import threading
from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.utils import embedding_functions
import requests

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/var/data/chroma")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

_client = None
_pdf_collection = None
_lock = threading.Lock()

def _init():
    global _client, _pdf_collection
    if _client:
        return

    with _lock:
        if _client:
            return

        print("🔄 Initializing Chroma...", flush=True)

        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        class OpenRouterEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, api_key, model_name="text-embedding-3-small"):
                self.api_key = api_key
                self.model_name = model_name
                self.url = "https://openrouter.ai/api/v1/embeddings"

            def __call__(self, input):
                if isinstance(input, str):
                    input = [input]
                response = requests.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"model": self.model_name, "input": input}
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]

        embed_fn = OpenRouterEmbeddingFunction(
            api_key=OPENROUTER_API_KEY,
            model_name="text-embedding-3-small"
        )

        _pdf_collection = _client.get_or_create_collection(
            name="pdf_embeddings",
            embedding_function=embed_fn
        )

        print("✅ Chroma ready", flush=True)

def query_pdfs(query: str, n: int = 5):
    _init()
    return _pdf_collection.query(query_texts=[query], n_results=n)

def get_pdf_collection():
    _init()
    return _pdf_collection
