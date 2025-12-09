import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import fitz  # PyMuPDF
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Add parent directory to path to import OpenRouter functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Parse'))

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")

if not API_KEY:
    print("❌ OPENROUTER_API_KEY not set in .env")
    exit()

app = Flask(__name__)
CORS(app)

# PDF folder path
PDF_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'Parse', 'pdfs')

# Global storage for processed documents
processed_documents = {
    'chunks': [],
    'embeddings': None,
    'is_processing': False,
    'is_ready': False
}

# Temporary storage for user-uploaded PDFs (session-based, not persisted)
temp_documents = {
    'chunks': [],
    'embeddings': None,
    'is_processing': False,
    'is_ready': False
}

def get_embedding_openrouter(text, model="text-embedding-3-large", verbose=False):
    """Get embeddings from OpenRouter API"""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "input": text}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if not response.text.strip():
            if verbose:
                print("❌ Empty response from OpenRouter embeddings endpoint")
            return np.zeros(3072)

        if response.status_code != 200:
            if verbose:
                print(f"❌ HTTP {response.status_code}: {response.text}")
            return np.zeros(3072)

        try:
            data = response.json()
        except Exception as e:
            if verbose:
                print("❌ Failed to parse JSON from response:", e)
                print("Response text:", response.text[:500])
            return np.zeros(3072)

        if "data" in data and len(data["data"]) > 0:
            return np.array(data["data"][0]["embedding"])
        else:
            if verbose:
                print("❌ Unexpected embedding response:", data)
            return np.zeros(3072)

    except Exception as e:
        if verbose:
            print(f"❌ Request failed: {e}")
        return np.zeros(3072)

def load_pdfs_on_startup():
    """Load and process PDFs from the pdfs folder on startup"""
    print("\n🚀 Loading PDFs from folder...")
    processed_documents['is_processing'] = True

    # Get all PDF files from the folder
    pdf_paths = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        print("❌ No PDF files found in folder.")
        processed_documents['is_processing'] = False
        return

    print(f"✅ Found {len(pdf_paths)} PDF(s)")

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

    # Generate embeddings (multi-threaded)
    def embed_chunk(i, chunk):
        emb = get_embedding_openrouter(chunk)
        if emb.shape[0] != 3072:
            if emb.shape[0] > 3072:
                emb = emb[:3072]
            else:
                emb = np.pad(emb, (0, 3072 - emb.shape[0]))
        return i, emb

    embeddings = []
    print("⚙️ Generating embeddings (multi-threaded)...")
    with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
        futures = [executor.submit(embed_chunk, i, chunk) for i, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            i, emb = future.result()
            embeddings.append((i, emb))
            print(f"  ✅ Chunk {i+1}/{len(chunks)} embedding done")

    embeddings.sort(key=lambda x: x[0])
    embeddings = np.stack([emb for _, emb in embeddings])

    # Store processed data
    processed_documents['chunks'] = chunks
    processed_documents['embeddings'] = embeddings
    processed_documents['is_processing'] = False
    processed_documents['is_ready'] = True

    print("✅ All PDFs processed and ready for chat!\n")

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    data = request.json

    if 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    user_query = data['query']

    # Combine permanent and temporary documents
    all_chunks = processed_documents['chunks'] + temp_documents['chunks']

    # Combine embeddings if both exist
    if processed_documents['embeddings'] is not None and temp_documents['embeddings'] is not None:
        all_embeddings = np.vstack([processed_documents['embeddings'], temp_documents['embeddings']])
    elif processed_documents['embeddings'] is not None:
        all_embeddings = processed_documents['embeddings']
    elif temp_documents['embeddings'] is not None:
        all_embeddings = temp_documents['embeddings']
    else:
        all_embeddings = None

    # Check if documents are processed
    if not all_chunks or all_embeddings is None:
        return jsonify({'error': 'No documents processed yet. Please upload PDFs first.'}), 400

    try:
        # Embed the user query
        q_emb = get_embedding_openrouter(user_query)

        if np.all(q_emb == 0):
            print("⚠️ Warning: Query embedding failed. Answers may be inaccurate.")

        # Retrieve top-k relevant chunks
        similarities = cosine_similarity([q_emb], all_embeddings)[0]
        top_k = min(5, len(all_chunks))
        top_indices = similarities.argsort()[-top_k:][::-1]
        retrieved_chunks = [all_chunks[i] for i in top_indices]
        context_text = "\n\n".join(retrieved_chunks)

        # Call OpenRouter chat API
        chat_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [
            {
                "role": "user",
                "content": f"Answer the question using only the following context:\n\n{context_text}\n\nQuestion: {user_query}"
            }
        ]
        payload = {
            "model": "openai/gpt-4o",
            "messages": messages,
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

@app.route('/api/upload-temp', methods=['POST'])
def upload_temp():
    """Handle temporary PDF uploads (not saved to database)"""
    if 'pdfs' not in request.files:
        return jsonify({'error': 'No PDF files provided'}), 400

    files = request.files.getlist('pdfs')

    if not files:
        return jsonify({'error': 'No files selected'}), 400

    print(f"\n📤 Received {len(files)} temporary PDF(s) for upload")
    temp_documents['is_processing'] = True

    try:
        documents = []

        # Process each uploaded PDF
        for file in files:
            if file.filename == '':
                continue

            if not file.filename.lower().endswith('.pdf'):
                continue

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name

            try:
                # Extract text from PDF
                doc = fitz.open(temp_path)
                text = "".join([page.get_text() + "\n" for page in doc])
                doc.close()

                if text.strip():
                    documents.append(text)
                    print(f"✅ Extracted text from: {file.filename} ({len(text)} chars)")
                else:
                    print(f"⚠️ No text found in: {file.filename}")

            except Exception as e:
                print(f"❌ Failed to read {file.filename}: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if not documents:
            temp_documents['is_processing'] = False
            return jsonify({'error': 'No text extracted from any PDFs'}), 400

        # Split text into chunks
        chunk_size = 500
        chunks = [doc[i:i + chunk_size] for doc in documents for i in range(0, len(doc), chunk_size)]
        print(f"📚 Created {len(chunks)} text chunks from temporary uploads.")

        # Generate embeddings
        def embed_chunk(i, chunk):
            emb = get_embedding_openrouter(chunk)
            if emb.shape[0] != 3072:
                if emb.shape[0] > 3072:
                    emb = emb[:3072]
                else:
                    emb = np.pad(emb, (0, 3072 - emb.shape[0]))
            return i, emb

        embeddings = []
        print("⚙️ Generating embeddings for temporary PDFs...")
        with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
            futures = [executor.submit(embed_chunk, i, chunk) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                i, emb = future.result()
                embeddings.append((i, emb))
                print(f"  ✅ Chunk {i+1}/{len(chunks)} embedding done")

        embeddings.sort(key=lambda x: x[0])
        embeddings = np.stack([emb for _, emb in embeddings])

        # Store in temporary documents
        temp_documents['chunks'] = chunks
        temp_documents['embeddings'] = embeddings
        temp_documents['is_processing'] = False
        temp_documents['is_ready'] = True

        print("✅ Temporary PDFs processed successfully!\n")

        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(files)} PDF(s)',
            'chunks': len(chunks)
        })

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        temp_documents['is_processing'] = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-temp', methods=['POST'])
def clear_temp():
    """Clear temporary uploaded documents"""
    temp_documents['chunks'] = []
    temp_documents['embeddings'] = None
    temp_documents['is_processing'] = False
    temp_documents['is_ready'] = False
    print("🗑️ Cleared temporary documents")
    return jsonify({'success': True, 'message': 'Temporary documents cleared'})

if __name__ == '__main__':
    # Load PDFs on startup in a background thread
    import threading
    thread = threading.Thread(target=load_pdfs_on_startup)
    thread.daemon = True
    thread.start()

    app.run(debug=True, port=5001, use_reloader=False)
