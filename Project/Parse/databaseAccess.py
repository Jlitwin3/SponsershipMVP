import os
import fitz
import numpy as np
from io import BytesIO
import dropbox
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import threading
# =========================================
# 0. Skip Dropbox indexing if no new embeddings
# =========================================
SKIP_DROPBOX_INDEXING = True
# =========================================
# 1. Load environment variables
# =========================================
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)
if not env_path: 
    raise ValueError("no env path")

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")

if not DROPBOX_ACCESS_TOKEN:
    raise ValueError("Missing required DropBox access token in .env")
if not API_KEY:
    raise ValueError("Missing or invalid api_key")

chat_history = []
# =========================================
# 2. Dropbox Setup
# =========================================
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
DROPBOX_FOLDER = "/L’mu-Oa (Sports Sponsorship AI Project)"
 # ← Dropbox folder path
res = dbx.files_list_folder('', recursive=True)
for entry in res.entries:
    print(entry.path_display)

# =========================================
# 3. ChromaDB Setup (Persistent)
# =========================================
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("pdf_embeddings")

# =========================================
# 4. Flask Setup
# =========================================
app = Flask(__name__)
CORS(app)

processing_status = {"is_processing": False, "is_ready": False}

# =========================================
# 5. Helper Functions
# =========================================
def fetch_dropbox_pdfs(folder_path=DROPBOX_FOLDER):
    """Get list of PDFs in Dropbox folder (recursively, including subfolders)."""
    print(f" Scanning Dropbox folder: {folder_path}")

    pdf_files = []

    try:
        result = dbx.files_list_folder(folder_path, recursive=True)
    except dropbox.exceptions.ApiError as e:
        print("Dropbox API error:", e)
        return pdf_files

    # Process first batch
    for entry in result.entries:
        if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith(".pdf"):
            pdf_files.append(entry)

    # Continue paging if there are more results
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith(".pdf"):
                pdf_files.append(entry)

    print(f" Found {len(pdf_files)} PDF(s).")
    return pdf_files



def extract_text_from_pdf(file_metadata):
    """Download and extract text from a Dropbox PDF"""
    _, res = dbx.files_download(file_metadata.path_lower)
    pdf_bytes = BytesIO(res.content)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = "".join(page.get_text() for page in doc)
    return text


def chunk_text(text, size=500):
    """Split text into smaller chunks"""
    return [text[i:i+size] for i in range(0, len(text), size)]


# =========================================
# 6. Embedding Function
# =========================================
openrouter_embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name="text-embedding-3-large"
)

# =========================================
# 7. Index PDFs Once
# =========================================
def get_all_ids(collection):
    """Fetch all existing document IDs from ChromaDB."""
    all_ids = []
    offset = 0
    batch_size = 100
    while True:
        result = collection.get(limit=batch_size, offset=offset)
        if not result["ids"]:
            break
        all_ids.extend(result["ids"])
        offset += batch_size
    return set(all_ids)


def index_pdfs_if_new():
    """Index only new or modified PDFs."""
    processing_status["is_processing"] = True
    pdf_files = fetch_dropbox_pdfs()

    # ✅ Correctly fetch existing IDs
    existing_ids = get_all_ids(collection)
    print("length of ids: ", len(existing_ids))

    for file_metadata in pdf_files:
        pdf_id = file_metadata.id or file_metadata.name

        # ✅ Check if this file (by name) already exists in DB
        existing = collection.get(where={"source": file_metadata.name})
        if existing["metadatas"]:
            existing_rev = existing["metadatas"][0].get("rev")
            # Skip if same revision (unchanged file)
            if existing_rev == file_metadata.rev:
                print(f"⏭️ Skipping {file_metadata.name} — no changes detected.")
                continue

        # ✅ Skip if already indexed and no revision check available
        if pdf_id in existing_ids:
            print(f"⏭️ Already indexed: {file_metadata.name}")
            continue

        # Otherwise, process and index
        print(f"📄 Processing: {file_metadata.name}")
        try:
            text = extract_text_from_pdf(file_metadata)
            chunks = chunk_text(text)

            if not chunks:
                print(f"⚠️ Skipping {file_metadata.name} — no text extracted.")
                continue

            base_id = pdf_id if pdf_id else file_metadata.name.replace(" ", "_").replace(".pdf", "")
            ids = [f"{base_id}_chunk{i}" for i in range(len(chunks))]
            ids = [i if i else f"{base_id}_chunk{idx}" for idx, i in enumerate(ids)]

            # ✅ Include revision info in metadata
            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=[{"source": file_metadata.name, "rev": file_metadata.rev}] * len(chunks)
            )

            print(f"✅ Indexed {len(chunks)} chunks from {file_metadata.name}")

        except Exception as e:
            print(f"❌ Failed to process {file_metadata.name}: {e}")

    processing_status["is_processing"] = False
    processing_status["is_ready"] = True
    print("📚 All Dropbox PDFs processed and stored in ChromaDB!")

# =========================================
# 8. Query ChromaDB and Chat via OpenRouter
# =========================================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    user_query = data["query"]

    if not processing_status["is_ready"]:
        return jsonify({"error": "PDFs still processing or not indexed yet."}), 400

    try:
        results = collection.query(query_texts=[user_query], n_results=5)
        if not results["documents"][0]:
            return jsonify({"answer": "No relevant context found."})

        context = "\n\n".join(results["documents"][0])
        if len(chat_history) > 10: 
            chat_history.pop(0)
        
        chat_history.append({"role": "user", "content": user_query})

        # Build context-aware conversation
        messages = [{"role": "system", "content": (
            "You are an expert research assistant that provides detailed, "
            "well-structured, and insightful responses. "
            "Always explain reasoning, give relevant examples, and cite context if available. If "
            "you are referencing a link, always convert it to a hyperlink"

        )}]
        messages += chat_history  # prior exchanges
        messages.append({
            "role": "user",
            "content": f"Relevant context:\n{context}\n\nNow answer the latest question above using this context."
        })

        payload = {
            "model": "openai/gpt-4o",
            "messages": messages,
            "extra_headers": {"HTTP-Referer": SITE_URL, "X-Title": SITE_TITLE}
        }
        

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        print("Openrouter response: ", r.status_code)
        data = r.json()
        print("Response JSON", data)

        if "choices" in data:
            answer = data["choices"][0]["message"]["content"]
            return jsonify({"answer": answer})
        elif "error" in data:
            return jsonify({"error": data["error"]["message"]}), 500
        else:
            return jsonify({"error": "Unexpected response from OpenRouter"}), 500

    except Exception as e:
        import traceback
        print(f"Full error traceback: ")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "processed": processing_status["is_ready"],
        "is_processing": processing_status["is_processing"],
        "collection_size": collection.count()
    })


# =========================================
# 9. Startup
# =========================================
if __name__ == "__main__":
    if not SKIP_DROPBOX_INDEXING:
        thread = threading.Thread(target=index_pdfs_if_new)
        thread.daemon = True
        thread.start()
        print("Dropbox indexing enabled. PDFs will be scanned and updated.")
    else:
        processing_status["is_ready"] = True
        print("Skipping Dropbox indexing — using existing ChromaDB data only.")

    print("\n Flask server running at http://localhost:5001")
    app.run(debug=True, port=5001, use_reloader=False)
