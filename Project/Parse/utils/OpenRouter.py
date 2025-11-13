import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import fitz  # PyMuPDF
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# -----------------------------
# 1. Load environment variables
# -----------------------------
# Try to load .env from current directory or parent Webapp directory
env_path = os.path.join(os.path.dirname(__file__), '.env')

if not os.path.exists(env_path):
    # Try parent Webapp directory
    env_path = os.path.join(os.path.dirname(__file__), '..', 'Webapp', '.env')
load_dotenv(env_path)
API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")

if not API_KEY:
    print("❌ OPENROUTER_API_KEY not set in .env")
    exit()

# -----------------------------
# 2. Initialize Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# 3. Global storage for processed documents
# -----------------------------
pdf_folder = os.path.join(os.path.dirname(__file__), "pdfs")
processed_documents = {
    'chunks': [],
    'embeddings': None,
    'is_processing': False,
    'is_ready': False
}

# -----------------------------
# 4. OpenRouter Embeddings
# -----------------------------
def get_embedding_openrouter(text, model="text-embedding-3-large"):
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "input": text}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return np.zeros(3072)
        data = response.json()
        if "data" in data and data["data"]:
            return np.array(data["data"][0]["embedding"])
    except:
        pass
    return np.zeros(3072)

# -----------------------------
# 5. Safe embedding wrapper
# -----------------------------
def safe_embed_texts(texts, embed_func, expected_dim=3072):
    embeddings = []
    print("⚙️ Generating embeddings (multi-threaded)...")
    def worker(i, t):
        emb = embed_func(t)
        if emb.shape[0] != expected_dim:
            if emb.shape[0] > expected_dim: 
                emb = emb[:expected_dim]
            else:
                emb = np.pad(emb, (0, expected_dim - emb.shape[0]))
        return i, emb

    with ThreadPoolExecutor(max_workers=min(10, len(texts))) as executor:
        futures = [executor.submit(worker, i, t) for i, t in enumerate(texts)]
        for future in as_completed(futures):
            i, emb = future.result()
            embeddings.append((i, emb))

    embeddings.sort(key=lambda x: x[0])
    return np.stack([emb for _, emb in embeddings])

# -----------------------------
# 6. Load and process PDFs on startup
# -----------------------------
def load_pdfs_on_startup():
    """Load and process PDFs from the pdfs folder and all subfolders"""
    print("\n🚀 Loading PDFs from folder (recursive)...")
    processed_documents['is_processing'] = True

    # Recursively get all PDF files
    pdf_paths = []
    for root, dirs, files in os.walk(pdf_folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, f))

    if not pdf_paths:
        print("❌ No PDF files found in folder or subfolders.")
        processed_documents['is_processing'] = False
        return

    print(f"✅ Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(" -", p)

    # Extract text from PDFs
    documents = []
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            text = "".join([page.get_text() + "\n" for page in doc])
            if text.strip():
                documents.append(text)
                print(f"✅ Extracted text from: {os.path.basename(pdf_path)} ({len(text)} chars)")
            else:
                print(f"⚠️ No text found in: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"❌ Failed to read {os.path.basename(pdf_path)}: {e}")

    if not documents:
        print("❌ No text extracted from any PDFs.")
        processed_documents['is_processing'] = False
        return

    # Split text into chunks
    chunk_size = 500
    chunks = [doc[i:i + chunk_size] for doc in documents for i in range(0, len(doc), chunk_size)]
    print(f"📚 Created {len(chunks)} text chunks.")

    # Generate all embeddings
    embeddings = safe_embed_texts(chunks, get_embedding_openrouter)

    # Store processed data
    processed_documents['chunks'] = chunks
    processed_documents['embeddings'] = embeddings
    processed_documents['is_processing'] = False
    processed_documents['is_ready'] = True

    print("\n✅ All PDFs processed and ready for chat!\n")
# -----------------------------
# 7. Flask API Routes
# -----------------------------
@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    data = request.json

    if 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    user_query = data['query']

    # Check if documents are processed
    if not processed_documents['chunks'] or processed_documents['embeddings'] is None:
        return jsonify({'error': 'No documents processed yet. Please wait for PDFs to be loaded.'}), 400

    try:
        # Embed the user query
        q_emb = get_embedding_openrouter(user_query)
        if np.all(q_emb == 0):
            print("⚠️ Warning: Query embedding failed. Answers may be inaccurate.")

        # Retrieve top-k relevant chunks
        similarities = cosine_similarity([q_emb], processed_documents['embeddings'])[0]
        top_k = min(5, len(processed_documents['chunks']))
        top_indices = similarities.argsort()[-top_k:][::-1]
        context = "\n\n".join(processed_documents['chunks'][i] for i in top_indices)

        # Call OpenRouter chat API
        chat_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "openai/gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": f"Answer using only this context:\n\n{context}\n\nQuestion: {user_query}"}
            ],
            "extra_headers": {"HTTP-Referer": SITE_URL, "X-Title": SITE_TITLE}
        }

        response = requests.post(chat_url, headers=headers, json=payload, timeout=60)

        try:
            data = response.json()
        except Exception as e:
            print("❌ Failed to parse chat JSON:", e)
            print("Response text:", response.text[:500])
            return jsonify({'error': 'Failed to parse chat response'}), 500

        if "choices" in data:
            answer = data["choices"][0]["message"]["content"]
            return jsonify({'answer': answer})
        elif "error" in data:
            return jsonify({'error': data["error"]["message"]}), 500
        else:
            return jsonify({'error': 'Unexpected response from API'}), 500

    except Exception as e:
        print(f"❌ Chat request failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check if documents are processed"""
    return jsonify({
        'processed': processed_documents['is_ready'],
        'is_processing': processed_documents['is_processing'],
        'chunks': len(processed_documents['chunks'])
    })

# -----------------------------
# 8. Start Flask server
# -----------------------------
if __name__ == '__main__':
    # Load PDFs on startup in a background thread
    thread = threading.Thread(target=load_pdfs_on_startup)
    thread.daemon = True
    thread.start()

    print("\n🌐 Starting Flask server on http://localhost:5001")
    print("📡 Frontend should connect to: http://localhost:5001")
    app.run(debug=True, port=5001, use_reloader=False)
