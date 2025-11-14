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
import re
from sponsorship.sponsor_manager import (
    check_sponsor_conflict,
    get_sponsor_info,
    get_all_current_sponsors,
    format_sponsor_context
)
from sponsorship.query_classifier import QueryClassifier
# =========================================
# 0. Skip Dropbox indexing if no new embeddings
# =========================================
SKIP_DROPBOX_INDEXING = True  # Enable to index PDFs from Dropbox
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
DROPBOX_FOLDER = "/L'mu-Oa (Sports Sponsorship AI Project)"  # Dropbox folder path

# =========================================
# 3. ChromaDB Setup (Persistent)
# =========================================
client = chromadb.PersistentClient(path="./chroma_db")

# Try to get existing collection first, create if it doesn't exist
try:
    collection = client.get_collection("pdf_embeddings")
    print(f"✅ PDF collection found. Current count: {collection.count()}")
except:
    # Collection doesn't exist, create it
    collection = client.create_collection(
        name="pdf_embeddings",
        metadata={"description": "PDF research papers"}
    )
    print(f"✅ Created new PDF collection")

# Get image collection (if it exists)
try:
    image_collection = client.get_collection("image_embeddings")
    print(f"✅ Image collection found with {image_collection.count()} embeddings")
except:
    image_collection = None
    print("⚠️  No image collection found - only PDFs will be searched")

# =========================================
# 4. Flask Setup
# =========================================
app = Flask(__name__)
CORS(app)

processing_status = {"is_processing": False, "is_ready": False}

# Initialize query classifier
query_classifier = QueryClassifier()

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
# 8. Helper Functions for Sponsor Detection
# =========================================
def extract_sponsor_mentions(query: str) -> list:
    """
    Extract potential sponsor names from query.
    Simple keyword matching - can be enhanced with NER later.
    """
    # Common sponsor keywords
    sponsor_keywords = ["propose", "sponsor", "partnership", "deal", "agreement"]
    query_lower = query.lower()

    # Check if this is a sponsor-related query
    if not any(kw in query_lower for kw in sponsor_keywords):
        return []

    # Try to extract company names (basic - looks for capitalized words)
    # This is a simple heuristic - can be improved
    words = query.split()
    potential_sponsors = []

    for i, word in enumerate(words):
        # Remove punctuation
        clean_word = re.sub(r'[^\w\s]', '', word)
        # Check if word is capitalized and not a common word
        if clean_word and clean_word[0].isupper() and clean_word.lower() not in ['can', 'we', 'should', 'would', 'what', 'how', 'why', 'when', 'where']:
            potential_sponsors.append(clean_word)

    return potential_sponsors

# =========================================
# 9. Query ChromaDB and Chat via OpenRouter
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
        # ===== Step 1: Classify the query =====
        classification = query_classifier.classify(user_query)
        print(f"📊 Query classified as: {classification['type']} (confidence: {classification['confidence']:.0%})")

        # Check if query is off-topic
        if classification['type'] == query_classifier.OFF_TOPIC:
            print("⚠️  Off-topic query detected - returning polite message")
            return jsonify({
                "answer": "I apologize, but I'm specifically designed to answer questions about sports sponsorships, particularly related to the University of Oregon's athletic partnerships. I don't have information to help with that topic. Feel free to ask me about sponsorship deals, brand partnerships, or anything related to sports marketing and sponsorships!"
            }), 200

        if classification['keywords']:
            print(f"   Keywords: {', '.join(classification['keywords'][:3])}")
        if classification['entities']:
            print(f"   Detected brands: {', '.join(classification['entities'])}")

        # ===== Step 2: Check sponsor database based on classification =====
        sponsor_context = ""
        sponsor_mentions = classification['entities'] if classification['needs_db'] else extract_sponsor_mentions(user_query)

        if sponsor_mentions:
            print(f"🔍 Detected potential sponsors in query: {sponsor_mentions}")

            for sponsor_name in sponsor_mentions:
                # Check for conflicts
                conflict_info = check_sponsor_conflict(sponsor_name)

                # Get detailed info if sponsor exists
                sponsor_info = get_sponsor_info(sponsor_name)

                if sponsor_info:
                    sponsor_context += format_sponsor_context(sponsor_info) + "\n"
                    print(f"✅ Found sponsor in database: {sponsor_name}")
                elif conflict_info['conflict']:
                    sponsor_context += f"\n[Important Sponsorship Notice]\n"
                    sponsor_context += f"Regarding {sponsor_name}:\n"
                    sponsor_context += f"{conflict_info['details']}\n"
                    if conflict_info['existing_sponsor']:
                        sponsor_context += f"Current Partner: {conflict_info['existing_sponsor']} ({conflict_info['category']})\n"
                    sponsor_context += "\n"
                    print(f"⚠️  Sponsor conflict detected: {sponsor_name}")

        # ===== Step 3: Query vector databases (adjust based on classification) =====
        # Adjust retrieval based on query type
        if classification['type'] == query_classifier.TEMPORAL:
            # For temporal queries, get more results to find recent ones
            pdf_count = 7
            image_count = 4
            print("   Using expanded search for temporal query")
        elif classification['type'] == query_classifier.LIST_REQUEST:
            # For list requests, get more diverse results
            pdf_count = 8
            image_count = 4
            print("   Using expanded search for list request")
        else:
            # Standard retrieval
            pdf_count = 5
            image_count = 3

        pdf_results = collection.query(query_texts=[user_query], n_results=pdf_count)

        # Initialize combined results
        all_documents = []
        all_metadatas = []

        # Add PDF results
        if pdf_results["documents"][0]:
            all_documents.extend(pdf_results["documents"][0])
            all_metadatas.extend(pdf_results["metadatas"][0])

        # Query image collection if it exists
        if image_collection is not None:
            try:
                image_results = image_collection.query(query_texts=[user_query], n_results=image_count)
                if image_results["documents"][0]:
                    all_documents.extend(image_results["documents"][0])
                    all_metadatas.extend(image_results["metadatas"][0])
                    print(f"✅ Found {len(image_results['documents'][0])} relevant image chunks")
            except Exception as img_err:
                print(f"⚠️  Error querying image collection: {img_err}")

        # Check if we have any results
        if not all_documents and not sponsor_context:
            return jsonify({"answer": "No relevant context found."})

        # Build context with source labels
        context_parts = []

        # Add sponsor database context first (highest priority)
        if sponsor_context:
            context_parts.append(sponsor_context)

        # Then add PDF and image results
        for doc, meta in zip(all_documents, all_metadatas):
            source_type = meta.get('type', 'pdf')
            source_name = meta.get('source', 'unknown')
            context_parts.append(f"[From {source_type}: {source_name}]\n{doc}")

        context = "\n\n".join(context_parts)
        if len(chat_history) > 10: 
            chat_history.pop(0)
        
        chat_history.append({"role": "user", "content": user_query})

        # Build context-aware conversation
        messages = [{"role": "system", "content": (
            "You are an expert sports sponsorship research assistant for University of Oregon Athletics. "
            "You provide detailed, well-structured, and insightful responses about sponsorships, partnerships, and sports business.\n\n"
            "IMPORTANT RULES:\n"
            "1. When 'Current UO Sponsor Information' is provided, treat it as authoritative official data\n"
            "2. When 'Important Sponsorship Notice' appears, clearly explain why conflicts exist\n"
            "3. Cite your sources naturally (e.g., 'According to current partnership records...' or 'Based on university documents...')\n"
            "4. When discussing existing sponsors, mention their exclusivity status if relevant\n"
            "5. If you reference a link, convert it to a hyperlink\n"
            "6. Be direct and clear about conflicts - students need accurate guidance\n"
            "7. Write in a professional but conversational tone - avoid overly technical database language\n\n"
            "Your goal is to educate students about UO sponsorship opportunities and constraints."
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
    image_count = image_collection.count() if image_collection else 0
    return jsonify({
        "processed": processing_status["is_ready"],
        "is_processing": processing_status["is_processing"],
        "collection_size": collection.count(),
        "image_collection_size": image_count,
        "total_documents": collection.count() + image_count
    })


@app.route("/api/sponsors", methods=["GET"])
def list_sponsors():
    """Get all current sponsors from the database."""
    try:
        sponsors = get_all_current_sponsors()
        return jsonify({
            "sponsors": sponsors,
            "count": len(sponsors)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sponsors/<sponsor_name>", methods=["GET"])
def get_sponsor_details(sponsor_name):
    """Get detailed information about a specific sponsor."""
    try:
        sponsor_info = get_sponsor_info(sponsor_name)
        if sponsor_info:
            return jsonify(sponsor_info)
        else:
            return jsonify({"error": f"Sponsor '{sponsor_name}' not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================
# 10. Startup
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
