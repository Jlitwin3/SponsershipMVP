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
    raise ValueError("Missing api_key")

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
client = chromadb.Client()
collection = client.create_collection(name="pdf_embeddings")

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
    """Get list of all PDFs in Dropbox folder (recursively, including subfolders)."""
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
def index_pdfs_if_new():
    """Index only new or unindexed PDFs"""
    processing_status["is_processing"] = True
    pdf_files = fetch_dropbox_pdfs()
    existing_ids = set(collection.get(ids=None)["ids"])

    for file_metadata in pdf_files:
        pdf_id = file_metadata.id
        if pdf_id in existing_ids:
            print(f"⏭️ Already indexed: {file_metadata.name}")
            continue

        print(f"📄 Processing: {file_metadata.name}")
        try:
            text = extract_text_from_pdf(file_metadata)
            chunks = chunk_text(text)
            
            # If there are no chunks, skip gracefully
            if not chunks:
                print(f"⚠️ Skipping {file_metadata.name} — no text extracted.")
                continue

            # Use filename as base ID if pdf_id is missing or empty
            base_id = pdf_id if pdf_id else file_metadata.name.replace(" ", "_").replace(".pdf", "")
            ids = [f"{base_id}_chunk{i}" for i in range(len(chunks))]

            # Safety check — ensure all IDs are unique & non-empty
            ids = [i if i else f"{base_id}_chunk{idx}" for idx, i in enumerate(ids)]

            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=[{"source": file_metadata.name}] * len(chunks)
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
            "Always explain reasoning, give relevant examples, and cite context if available."
            "always use the context provided unless it does not specify information related to the question"
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
        data = r.json()

        if "choices" in data:
            answer = data["choices"][0]["message"]["content"]
            return jsonify({"answer": answer})
        elif "error" in data:
            return jsonify({"error": data["error"]["message"]}), 500
        else:
            return jsonify({"error": "Unexpected response from OpenRouter"}), 500

    except Exception as e:
        print(f"❌ Chat request failed: {e}")
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
    thread = threading.Thread(target=index_pdfs_if_new)
    thread.daemon = True
    thread.start()

    print("\n🌐 Flask server running at http://localhost:5001")
    app.run(debug=True, port=5001, use_reloader=False)
