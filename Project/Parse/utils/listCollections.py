import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collections = client.list_collections()
for c in collections: 
    
    sample = c.get(limit=1, include=["documents", "metadatas"])
    if sample["documents"]: 
        print("sample docs: ", sample["documents"][:300])
        print("metadatas: ", sample["metadatas"])
    else: 
        print("no doocuments found")

