import chromadb

# Create a client using the new Settings object
client = chromadb.PersistentClient(path="./chroma_db")

# List collections to see what’s stored
collection = client.get_or_create_collection(name="pdf_embeddings")
#print("Collections:", [c.name for c in collections])
results = collection.get(
    include=['embeddings']
)
all_embeddings = results['embeddings']

# You can then iterate through or process these embeddings as needed
for embedding in all_embeddings:
    print(embedding)


