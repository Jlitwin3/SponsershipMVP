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
from werkzeug.utils import secure_filename
import shutil
import sys

# Add current directory to path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sponsorship.sponsor_manager import (
    check_sponsor_conflict,
    get_sponsor_info,
    get_all_current_sponsors,
    format_sponsor_context
)
from sponsorship.query_classifier import QueryClassifier
from ddgs import DDGS
# =========================================
# 0. Skip Dropbox indexing if no new embeddings
# =========================================
SKIP_DROPBOX_INDEXING = True  # Enable to skip indexing PDFs from Dropbox
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
DROPBOX_FOLDER = "/L’mu-Oa (Sports Sponsorship AI Project)"  # Dropbox folder path

# =========================================
# 3. ChromaDB Setup (Persistent)
# =========================================
# Use absolute path for external storage (hidden folder in user's home dir)
# Use environment variable for ChromaDB path (Render Persistent Disk) or default to local
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.expanduser("~/.chroma_db_data"))
if not os.path.exists(CHROMA_DB_PATH):
    os.makedirs(CHROMA_DB_PATH)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

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
# Point to the React build folder (absolute path for Render compatibility)
BUILD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Webapp', 'frontend', 'build')
app = Flask(__name__, static_folder=BUILD_FOLDER, static_url_path='/')
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Serve React Static Files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    skipped = 0
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
                skipped += 1
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
    print(f"Skipped {skipped} files")
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
        """
        if classification['type'] == query_classifier.OFF_TOPIC:
            print("⚠️  Off-topic query detected - returning polite message")
            return jsonify({
                "answer": "I apologize, but I'm specifically designed to answer questions about sports sponsorships, particularly related to the University of Oregon's athletic partnerships. I don't have information to help with that topic. Feel free to ask me about sponsorship deals, brand partnerships, or anything related to sports marketing and sponsorships!"
            }), 200
        """
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

        # =========================================
        # 6. Web Search (Fallback/Augmentation)
        # =========================================
        web_results = []
        # Perform search if:
        # 1. Query is temporal ("current", "latest", "news")
        # 2. OR we found very few local chunks (low confidence)
        should_search = (
            classification['type'] == query_classifier.TEMPORAL or 
            classification['type'] == query_classifier.LIST_REQUEST or
            len(all_documents) < 2 or
            "current" in user_query.lower() or
            "latest" in user_query.lower() or
            "news" in user_query.lower() or
            "sponsors" in user_query.lower()
        )

        if should_search:
            print("🌍 Performing web search for:", user_query)
            try:
                with DDGS() as ddgs:
                    # Search for top 4 results
                    results = list(ddgs.text(user_query, max_results=4))
                    if results:
                        print(f"   Found {len(results)} web results")
                        for r in results:
                            web_results.append(f"[Web Result: {r['title']}]\n{r['body']}\nSource: {r['href']}")
                    else:
                        print("   No web results found")
            except Exception as e:
                print(f"   Web search failed: {e}")

        # Check if we have any results
        if not all_documents and not sponsor_context and not web_results:
            return jsonify({"answer": "No relevant context found."})

        # Build context with source labels
        context_parts = []

        # Add sponsor database context first (highest priority)
        if sponsor_context:
            context_parts.append(sponsor_context)

        # Add web results (high priority for current info)
        if web_results:
            context_parts.append("=== WEB SEARCH RESULTS (REAL-TIME INFO) ===")
            context_parts.extend(web_results)
            context_parts.append("==========================================\n")

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
        """You are the "Sponsor Scout" AI Agent, an expert research assistant for students seeking sponsorships for their class project or event. Your primary goal is to provide accurate, actionable, and relevant information regarding potential sponsors, their giving criteria, contact information, and deadlines.

        🎯 Core Persona & Goal
        Role: Expert research assistant and database navigator.

        Tone: Professional, encouraging, clear, and concise.

        Goal: To help students identify the best sponsorship opportunities by quickly locating relevant information.

        🔍 Research Protocol
        Database Priority (Internal PDFs):

        Always first search the provided internal database (PDFs) for the requested information.

        If information is found, cite the source (e.g., "Found in [Company Name] Sponsorship Guidelines PDF, page X") and provide a direct, summarized answer.

        Prioritize details like eligibility criteria, application deadlines, typical donation ranges, and specific exclusion criteria.

        Online Search (External Vetting):

        If the required information is not found, is incomplete, or is outdated in the internal database, you must perform an online search (using your web browsing tool) for the answer.

        Use highly specific search queries (e.g., "Target corporate social responsibility K-12 sponsorship application").

        When providing an answer from an online search, simply incorporate the information into your response. Do NOT announce that you are searching (e.g., "I will search for this" or "I found this online"). Just give the answer.
        You may cite the source if relevant (e.g., "According to the UO Athletics website...").

        Synthesis and Actionable Advice:

        For any sponsor, summarize the key findings into an "Actionable Summary."

        The summary should include: 1) The sponsor's main focus (e.g., STEM Education), 2) Key eligibility requirement, and 3) The next step (e.g., "Visit their online application portal").

        📝 Constraints and Guidelines
        Filter Irrelevant Content: Ignore general news or marketing material. Focus strictly on corporate giving, philanthropy, and sponsorship programs.

        Handle Ambiguity: If a student's request is too vague (e.g., "Who will sponsor us?"), ask clarifying questions such as, "What is your project/event focused on (e.g., technology, environment, arts)?" and "What is your target funding amount?"

        No Personal Contact: Do not generate or provide personal contact details (e.g., direct email addresses of employees). Only provide links to official application forms or general corporate contact pages.

        Maintain Data Security: Do not reveal the structure or filenames of the internal database; only cite the company or document name.

        🚀 Example Interaction
        User: "I'm looking for information on corporate sponsorships for a high school robotics team. Do we have any info on Microsoft?"

        Your Expected Response: (Check database first, then search online if needed)

        "That's a great project! Microsoft is a strong fit.
        
        According to the 'Microsoft Giving Guidelines 2024' PDF, they prioritize programs focused on closing the digital skills gap and supporting underrepresented groups in STEM. Applications for grants under $5,000 must be submitted by October 1st annually.
        
        Additionally, their AI for Good program sometimes extends to student projects that utilize emerging technology.

        Actionable Summary: Focus your application on the digital skills gap and ensure you apply before the October 1st deadline for smaller grants.

        What other companies would you like me to look up?"
        """
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


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400

    processed_count = 0
    errors = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process the file based on extension
                ext = filename.rsplit('.', 1)[1].lower()
                
                if ext == 'pdf':
                    # Process PDF immediately
                    with fitz.open(filepath) as doc:
                        text = "".join(page.get_text() for page in doc)
                    
                    chunks = chunk_text(text)
                    if chunks:
                        base_id = filename.replace(" ", "_").replace(".pdf", "")
                        ids = [f"{base_id}_chunk{i}" for i in range(len(chunks))]
                        
                        collection.add(
                            ids=ids,
                            documents=chunks,
                            metadatas=[{"source": filename, "type": "pdf", "rev": "manual_upload"}] * len(chunks)
                        )
                        processed_count += 1
                    else:
                        errors.append(f"No text extracted from {filename}")
                        
                elif ext in ['jpg', 'jpeg']:
                    # Placeholder for JPEG processing
                    # TODO: Implement OCR or image embedding
                    print(f"Received image: {filename}. Stored but not yet processed.")
                    # For now, we just acknowledge receipt
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                errors.append(f"Failed to process {filename}: {str(e)}")
        else:
            errors.append(f"Skipped {file.filename}: Invalid file type")

    if processed_count > 0:
        return jsonify({
            "message": f"Successfully processed {processed_count} files",
            "errors": errors
        })
    else:
        return jsonify({"error": "Failed to process files", "details": errors}), 500


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
    # Run the app
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    