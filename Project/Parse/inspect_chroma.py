import chromadb
import os

CHROMA_DB_PATH = os.path.expanduser("~/.chroma_db_data")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

print(f"📂 Inspecting ChromaDB at: {CHROMA_DB_PATH}")
collections = client.list_collections()
print(f"Found {len(collections)} collections:")

for col in collections:
    print(f"\n--- Collection: {col.name} ---")
    print(f"  Count: {col.count()}")
    print(f"  Metadata: {col.metadata}")
    # Try to get one item to see embedding dimension if possible
    try:
        items = col.get(limit=1, include=['embeddings'])
        if items['embeddings']:
            print(f"  Embedding Dimension: {len(items['embeddings'][0])}")
        else:
            print("  No embeddings found in this collection.")
    except Exception as e:
        print(f"  Could not retrieve embeddings: {e}")
