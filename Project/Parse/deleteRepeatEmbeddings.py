import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_embeddings"  # change this to your collection name
SIMILARITY_THRESHOLD = 0.9999   # 1.0 = identical; use 0.9999 for near-duplicates

# --- CONNECT TO CHROMA ---
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(COLLECTION_NAME)

# --- FETCH ALL DATA ---
print(f"🔍 Fetching all embeddings from '{COLLECTION_NAME}'...")
all_data = collection.get(include=["embeddings", "documents", "metadatas"])
embeddings = np.array(all_data["embeddings"])
ids = all_data["ids"]
docs = all_data["documents"]

if len(embeddings) == 0:
    print("⚠️ No embeddings found in this collection.")
    exit()

print(f"✅ Retrieved {len(embeddings)} embeddings. Checking for duplicates...")

# --- FIND DUPLICATES ---
to_delete = set()
checked = set()

for i in range(len(embeddings)):
    if i in checked:
        continue
    # Compare current embedding with the rest
    sims = cosine_similarity([embeddings[i]], embeddings)[0]
    # Find indices of near-identical embeddings
    dup_indices = np.where(sims > SIMILARITY_THRESHOLD)[0]
    dup_indices = [idx for idx in dup_indices if idx != i]

    if dup_indices:
        print(f"🧩 Found {len(dup_indices)} duplicates of ID {ids[i]}")
        for j in dup_indices:
            to_delete.add(ids[j])
        checked.update(dup_indices)
    checked.add(i)

# --- DELETE DUPLICATES ---
if to_delete:
    print(f"🧹 Removing {len(to_delete)} duplicate embeddings...")
    collection.delete(ids=list(to_delete))
    print("✅ Duplicates removed successfully.")
else:
    print("✨ No duplicates found. Your collection is clean!")
