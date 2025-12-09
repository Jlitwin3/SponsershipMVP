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
import tempfile

# =========================================
# 1. Load environment variables (MUST BE FIRST)
# =========================================
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)
if not env_path:
    raise ValueError("no env path")

# Add current directory to path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sponsorship.sponsor_manager import (
    check_sponsor_conflict,
    get_sponsor_info,
    get_all_current_sponsors,
    format_sponsor_context
)
from sponsorship.query_classifier import QueryClassifier

from google import genai
from google.genai import types

# Initialize Gemini Client Globally
if os.getenv("GOOGLE_API_KEY"):
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
else:
    gemini_client = None
    grounding_tool = None
# =========================================
# 0. Skip Dropbox indexing if no new embeddings
# =========================================
SKIP_DROPBOX_INDEXING = os.getenv("SKIP_DROPBOX_INDEXING", "True").lower() == "true"  # Default to True to prevent auto-indexing on startup

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



if not DROPBOX_ACCESS_TOKEN:
    raise ValueError("Missing required DropBox access token in .env")
if not API_KEY:
    raise ValueError("Missing required OpenRouter API key in .env")
if not GOOGLE_API_KEY:
    print("⚠️  Warning: GOOGLE_API_KEY not found in .env - Gemini search will be disabled")

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
print(f"🔍 BUILD_FOLDER path: {BUILD_FOLDER}")
print(f"🔍 BUILD_FOLDER exists: {os.path.exists(BUILD_FOLDER)}")
if os.path.exists(BUILD_FOLDER):
    print(f"🔍 Files in BUILD_FOLDER: {os.listdir(BUILD_FOLDER)[:10]}")  # Show first 10 files
else:
    print(f"❌ BUILD_FOLDER not found! Current working directory: {os.getcwd()}")
    print(f"❌ __file__ location: {os.path.abspath(__file__)}")
    
app = Flask(__name__, static_folder=BUILD_FOLDER, static_url_path='/')
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize status - assume ready if skipping indexing
processing_status = {
    "is_processing": False, 
    "is_ready": SKIP_DROPBOX_INDEXING  # If skipping indexing, we are ready immediately
}

# Temporary storage for user-uploaded PDFs (session-based, not persisted to ChromaDB)
temp_documents = {
    'texts': [],  # Store raw text chunks
    'is_processing': False,
    'is_ready': False
}

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
    model_name="text-embedding-3-large",
    api_base="https://openrouter.ai/api/v1"
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
        
        # Web search disabled - relying on GPT-4's internal knowledge
        # print(f"🔍 Search decision: should_search={should_search}, classification={classification['type']}, doc_count={len(all_documents)}, has_sponsors={'sponsors' in user_query.lower()}")
        
        # if should_search and GOOGLE_API_KEY:
        #     print("🤖 Performing Gemini AI search for:", user_query)
        #     ... (search code commented out)

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

        # Add temporary uploaded documents if available
        if temp_documents['is_ready'] and temp_documents['texts']:
            print(f"📄 Including {len(temp_documents['texts'])} temporary document chunks")
            for temp_text in temp_documents['texts'][:5]:  # Limit to 5 chunks
                context_parts.append(f"[From temporary upload]\n{temp_text}")

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
        1. **Check Provided Context First**: Review any context from the internal database (PDFs, images) for relevant information.
        
        2. **MANDATORY: Use Google Search When Information is Incomplete**: 
           - If the provided context doesn't fully answer the question, you MUST use Google Search
           - ALWAYS use Google Search for queries containing: "current", "latest", "recent", "all", "complete list", "comprehensive"
           - When a user asks for "more", "complete", "whole list", or "all" - this means you MUST search Google to provide additional information
           - Do NOT say "I don't have" or "I cannot provide" - instead, USE GOOGLE SEARCH to find the answer
        
        3. **NEVER Cite Sources Unless Explicitly Asked**: 
           - Do NOT use [cite: ...] tags in your responses
           - Do NOT mention document names, PDFs, or where information came from
           - Do NOT say "according to" or "based on" unless the user specifically asks "what are your sources?"
           - Just provide the information naturally as if you know it
        
        4. **Answer Confidently Without Disclaimers**: 
           - NEVER say: "I don't have a comprehensive list"
           - NEVER say: "The provided documents do not contain this"
           - NEVER say: "I cannot find this in the database"
           - Instead: Use Google Search to find the answer and present it confidently


        📝 Constraints and Guidelines
        Filter Irrelevant Content: Ignore general news or marketing material. Focus strictly on corporate giving, philanthropy, and sponsorship programs.

        Handle Ambiguity: If a student's request is too vague (e.g., \"Who will sponsor us?\"), ask clarifying questions such as, \"What is your project/event focused on (e.g., technology, environment, arts)?\" and \"What is your target funding amount?\"

        No Personal Contact: Do not generate or provide personal contact details (e.g., direct email addresses of employees). Only provide links to official application forms or general corporate contact pages.

        Maintain Data Security: Do not reveal the structure or filenames of the internal database; only cite the company or document name.

        🚀 Example Interaction
        User: \"I'm looking for information on corporate sponsorships for a high school robotics team. Do we have any info on Microsoft?\"

        Your Expected Response: (Check database first, then use your knowledge)

        \"That's a great project! Microsoft is a strong fit.
        
        According to the 'Microsoft Giving Guidelines 2024' PDF, they prioritize programs focused on closing the digital skills gap and supporting underrepresented groups in STEM. Applications for grants under $5,000 must be submitted by October 1st annually.
        
        Additionally, their AI for Good program sometimes extends to student projects that utilize emerging technology.

        Actionable Summary: Focus your application on the digital skills gap and ensure you apply before the October 1st deadline for smaller grants.

        What other companies would you like me to look up?\"
        """
        )}]
        messages += chat_history  # prior exchanges
        messages.append({
            "role": "user",
            "content": f"{context}\n\nQuestion: {user_query}"
        })
        # Old gpt-4 model
        """
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
        """
        # Use Gemini for response generation with Google Search enabled
        if not gemini_client:
             return jsonify({"error": "Gemini API key not configured"}), 500

        config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )
        
        # Construct the full prompt from messages
        system_prompt = messages[0]['content']
        full_prompt = f"{system_prompt}\n\nChat History:\n"
        
        for msg in chat_history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"
            
        full_prompt += f"\nContext:\n{context}\n\nUser Question: {user_query}"

        print("DEBUG: Sending request to Gemini...")
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=full_prompt,
                config=config
            )
            print("DEBUG: Gemini response received.")
        except Exception as e:
            print(f"ERROR: Gemini generation failed: {e}")
            raise e
        
        if response.text:
            return jsonify({"answer": response.text})
        else:
            return jsonify({"error": "No response generated by Gemini"}), 500

    except Exception as e:
        import traceback
        print(f"Full error traceback: ")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    # Optimize status check - avoid expensive count() calls on every poll
    # Just return the status flags
    return jsonify({
        "processed": processing_status["is_ready"],
        "is_processing": processing_status["is_processing"],
        "collection_size": "11000+", # Hardcoded for performance or cache this
        "image_collection_size": "2000+",
        "total_documents": "13000+"
    })

"""
@app.route("/api/sponsors", methods=["GET"])
def list_sponsors():
    ""Get all current sponsors from the database.""
    try:
        sponsors = get_all_current_sponsors()
        return jsonify({
            "sponsors": sponsors,
            "count": len(sponsors)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""
@app.route("/api/admin/upload", methods=["POST"])
def upload_file():
    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('files')

    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400

    processed_count = 0
    errors = []
    uploaded_files = []

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
                        # Add timestamp to ensure unique IDs even for duplicate filenames
                        from datetime import datetime
                        upload_date = datetime.now().isoformat()
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]  # Include microseconds

                        base_id = f"admin_{timestamp}_{filename.replace(' ', '_').replace('.pdf', '')}"
                        ids = [f"{base_id}_chunk{i}" for i in range(len(chunks))]

                        collection.add(
                            ids=ids,
                            documents=chunks,
                            metadatas=[{
                                "source": filename,
                                "type": "pdf",
                                "upload_type": "admin_upload",
                                "upload_date": upload_date,
                                "file_size": os.path.getsize(filepath)
                            }] * len(chunks)
                        )
                        processed_count += 1
                        uploaded_files.append({
                            "name": filename,
                            "size": os.path.getsize(filepath),
                            "chunks": len(chunks),
                            "date": upload_date
                        })
                        print(f"✅ Uploaded and indexed {filename} ({len(chunks)} chunks)")
                    else:
                        errors.append(f"No text extracted from {filename}")

                elif ext in ['jpg', 'jpeg']:
                    # Placeholder for JPEG processing - store with admin_upload tag
                    from datetime import datetime
                    upload_date = datetime.now().isoformat()

                    # For now, just store metadata without processing
                    # TODO: Implement OCR with Tesseract or Google Vision API
                    print(f"Received image: {filename}. Stored but OCR not yet implemented.")
                    processed_count += 1
                    uploaded_files.append({
                        "name": filename,
                        "size": os.path.getsize(filepath),
                        "chunks": 0,
                        "date": upload_date,
                        "note": "Image stored, OCR processing pending"
                    })

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                errors.append(f"Failed to process {filename}: {str(e)}")
        else:
            errors.append(f"Skipped {file.filename}: Invalid file type")

    if processed_count > 0:
        return jsonify({
            "message": f"Successfully processed {processed_count} file(s)",
            "files": uploaded_files,
            "errors": errors
        })
    else:
        return jsonify({"error": "Failed to process files", "details": errors}), 500


@app.route("/api/admin/files", methods=["GET"])
def get_admin_files():
    """Get all admin-uploaded files from ChromaDB."""
    try:
        # Query ChromaDB for documents with upload_type = "admin_upload"
        results = collection.get(
            where={"upload_type": "admin_upload"},
            include=["metadatas"]
        )

        # Group by filename to avoid duplicates (since we have chunks)
        files_dict = {}
        for metadata in results["metadatas"]:
            filename = metadata.get("source")
            if filename and filename not in files_dict:
                # Convert bytes to MB
                file_size_bytes = metadata.get("file_size", 0)
                file_size_mb = round(file_size_bytes / (1024 * 1024), 2) if file_size_bytes else 0

                files_dict[filename] = {
                    "name": filename,
                    "size": f"{file_size_mb} MB",
                    "type": metadata.get("type", "unknown").upper(),
                    "date": metadata.get("upload_date", "Unknown")[:10]  # Just date part
                }

        # Convert to list
        files_list = list(files_dict.values())

        return jsonify({
            "files": files_list,
            "count": len(files_list)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/delete", methods=["POST"])
def delete_file():
    """Delete an admin-uploaded file from ChromaDB."""
    try:
        data = request.json
        if not data or "filename" not in data:
            return jsonify({"error": "No filename provided"}), 400

        filename = data["filename"]
        confirm_filename = data.get("confirmFilename", "")

        # Verify filenames match
        if filename != confirm_filename:
            return jsonify({"error": "Filenames do not match"}), 400

        # Query for all chunks of this file with upload_type = "admin_upload"
        results = collection.get(
            where={
                "$and": [
                    {"source": filename},
                    {"upload_type": "admin_upload"}
                ]
            },
            include=["metadatas"]
        )

        if not results["ids"]:
            return jsonify({"error": f"File '{filename}' not found in admin uploads"}), 404

        chunk_count = len(results["ids"])

        # Delete all chunks
        collection.delete(ids=results["ids"])

        # Also delete the physical file if it exists
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🗑️ Deleted physical file: {filepath}")

        print(f"✅ Deleted {chunk_count} chunks for file: {filename}")

        return jsonify({
            "message": f"Successfully deleted '{filename}' ({chunk_count} chunks)",
            "chunks_deleted": chunk_count
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
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

@app.route("/api/admin/reindex", methods=["POST"])
def trigger_reindex():
    """Manually trigger Dropbox re-indexing."""
    if processing_status.get("is_processing"):
        return jsonify({"message": "Indexing already in progress"}), 409
    
    thread = threading.Thread(target=index_pdfs_if_new)
    thread.daemon = True
    thread.start()
    return jsonify({"message": "Started background indexing process"})

@app.route("/api/admin/reset_db", methods=["POST"])
def reset_db():
    """Delete existing collections and start fresh. Use this if embedding dimensions are mismatched."""
    try:
        try:
            client.delete_collection("pdf_embeddings")
            print("🗑️ Deleted pdf_embeddings collection")
        except:
            pass
            
        try:
            client.delete_collection("image_embeddings")
            print("🗑️ Deleted image_embeddings collection")
        except:
            pass
            
        # Re-create collections with correct embedding function
        global collection, image_collection
        collection = client.create_collection(
            name="pdf_embeddings",
            embedding_function=openrouter_embed,
            metadata={"description": "PDF research papers"}
        )
        
        image_collection = client.create_collection(
            name="image_embeddings",
            embedding_function=openrouter_embed
        )
        
        # Trigger re-indexing
        thread = threading.Thread(target=index_pdfs_if_new)
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": "Database reset and re-indexing started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/verify-email", methods=["POST"])
def verify_email():
    """Verify if an email is whitelisted for admin access."""
    try:
        data = request.json
        email = data.get("email", "").strip().lower()

        if not email:
            return jsonify({"error": "Email is required", "authorized": False}), 400

        # Get ADMIN list from environment variable
        admin_str = os.getenv("ADMIN_LIST", "")
        admin_emails = [e.strip().lower() for e in admin_str.split(",") if e.strip()]

        # Check if email is in admin list
        is_authorized = email in admin_emails

        print(f"🔍 Admin access attempt: {email} - {'✅ Authorized' if is_authorized else '❌ Denied'}")

        if is_authorized:
            return jsonify({"authorized": True, "message": "Access granted"})
        else:
            return jsonify({"authorized": False, "message": "Access denied"}), 403

    except Exception as e:
        print(f"❌ ERROR in verify_email: {e}")
        return jsonify({"error": str(e), "authorized": False}), 500

@app.route("/api/chatbot/verify-email", methods=["POST"])
def verify_chatbot_email():
    """Verify if an email is whitelisted for chatbot access (uses USER_WHITELIST)."""
    try:
        data = request.json
        email = data.get("email", "").strip().lower()

        if not email:
            return jsonify({"error": "Email is required", "authorized": False}), 400

        # Get USER whitelist from environment variable
        user_whitelist_str = os.getenv("USER_WHITELIST", "")
        whitelisted_emails = [e.strip().lower() for e in user_whitelist_str.split(",") if e.strip()]

        # Also get admin list - admins can access chatbot too
        admin_str = os.getenv("ADMIN_LIST", "")
        admin_emails = [e.strip().lower() for e in admin_str.split(",") if e.strip()]

        # Check if email is in either list
        is_authorized = email in whitelisted_emails or email in admin_emails

        print(f"🔍 Chatbot access attempt: {email} - {'✅ Authorized' if is_authorized else '❌ Denied'}")

        if is_authorized:
            return jsonify({"authorized": True, "message": "Access granted"})
        else:
            return jsonify({"authorized": False, "message": "Access denied"}), 403

    except Exception as e:
        print(f"❌ ERROR in verify_chatbot_email: {e}")
        return jsonify({"error": str(e), "authorized": False}), 500

@app.route("/api/admin/whitelist", methods=["GET"])
def get_whitelist():
    """Get the current user email whitelist."""
    try:
        whitelist_str = os.getenv("USER_WHITELIST", "")
        whitelisted_emails = [e.strip() for e in whitelist_str.split(",") if e.strip()]

        return jsonify({
            "emails": whitelisted_emails,
            "count": len(whitelisted_emails)
        })
    except Exception as e:
        print(f"❌ ERROR in get_whitelist: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/whitelist/add", methods=["POST"])
def add_to_whitelist():
    """Add an email to the user whitelist (updates .env file)."""
    try:
        data = request.json
        new_email = data.get("email", "").strip().lower()

        if not new_email:
            return jsonify({"error": "Email is required"}), 400

        # Basic email validation
        import re
        email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_regex, new_email):
            return jsonify({"error": "Invalid email format"}), 400

        # Get current whitelist
        whitelist_str = os.getenv("USER_WHITELIST", "")
        whitelisted_emails = [e.strip().lower() for e in whitelist_str.split(",") if e.strip()]

        # Check if already exists
        if new_email in whitelisted_emails:
            return jsonify({"error": "Email already in whitelist"}), 400

        # Add new email
        whitelisted_emails.append(new_email)
        new_whitelist_str = ",".join(whitelisted_emails)

        # Update .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        update_env_file(env_path, "USER_WHITELIST", new_whitelist_str)

        # Update environment variable for current process
        os.environ["USER_WHITELIST"] = new_whitelist_str

        print(f"✅ Added {new_email} to user whitelist")

        return jsonify({
            "message": f"Successfully added {new_email} to user whitelist",
            "emails": whitelisted_emails
        })
    except Exception as e:
        print(f"❌ ERROR in add_to_whitelist: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/whitelist/remove", methods=["POST"])
def remove_from_whitelist():
    """Remove an email from the user whitelist (updates .env file)."""
    try:
        data = request.json
        email_to_remove = data.get("email", "").strip().lower()

        if not email_to_remove:
            return jsonify({"error": "Email is required"}), 400

        # Get current whitelist
        whitelist_str = os.getenv("USER_WHITELIST", "")
        whitelisted_emails = [e.strip().lower() for e in whitelist_str.split(",") if e.strip()]

        # Check if exists
        if email_to_remove not in whitelisted_emails:
            return jsonify({"error": "Email not found in whitelist"}), 404

        # Remove email
        whitelisted_emails.remove(email_to_remove)
        new_whitelist_str = ",".join(whitelisted_emails)

        # Update .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        update_env_file(env_path, "USER_WHITELIST", new_whitelist_str)

        # Update environment variable for current process
        os.environ["USER_WHITELIST"] = new_whitelist_str

        print(f"✅ Removed {email_to_remove} from user whitelist")

        return jsonify({
            "message": f"Successfully removed {email_to_remove} from user whitelist",
            "emails": whitelisted_emails
        })
    except Exception as e:
        print(f"❌ ERROR in remove_from_whitelist: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/adminlist", methods=["GET"])
def get_adminlist():
    """Get the current admin list."""
    try:
        admin_str = os.getenv("ADMIN_LIST", "")
        admin_emails = [e.strip() for e in admin_str.split(",") if e.strip()]

        return jsonify({
            "emails": admin_emails,
            "count": len(admin_emails)
        })
    except Exception as e:
        print(f"❌ ERROR in get_adminlist: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/adminlist/add", methods=["POST"])
def add_to_adminlist():
    """Add an email to the admin list (updates .env file)."""
    try:
        data = request.json
        new_email = data.get("email", "").strip().lower()

        if not new_email:
            return jsonify({"error": "Email is required"}), 400

        # Basic email validation
        import re
        email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_regex, new_email):
            return jsonify({"error": "Invalid email format"}), 400

        # Get current admin list
        admin_str = os.getenv("ADMIN_LIST", "")
        admin_emails = [e.strip().lower() for e in admin_str.split(",") if e.strip()]

        # Check if already exists
        if new_email in admin_emails:
            return jsonify({"error": "Email already in admin list"}), 400

        # Add new email
        admin_emails.append(new_email)
        new_admin_str = ",".join(admin_emails)

        # Update .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        update_env_file(env_path, "ADMIN_LIST", new_admin_str)

        # Update environment variable for current process
        os.environ["ADMIN_LIST"] = new_admin_str

        print(f"✅ Added {new_email} to admin list")

        return jsonify({
            "message": f"Successfully added {new_email} to admin list",
            "emails": admin_emails
        })
    except Exception as e:
        print(f"❌ ERROR in add_to_adminlist: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/adminlist/remove", methods=["POST"])
def remove_from_adminlist():
    """Remove an email from the admin list (updates .env file)."""
    try:
        data = request.json
        email_to_remove = data.get("email", "").strip().lower()

        if not email_to_remove:
            return jsonify({"error": "Email is required"}), 400

        # Get current admin list
        admin_str = os.getenv("ADMIN_LIST", "")
        admin_emails = [e.strip().lower() for e in admin_str.split(",") if e.strip()]

        # Check if exists
        if email_to_remove not in admin_emails:
            return jsonify({"error": "Email not found in admin list"}), 404

        # Remove email
        admin_emails.remove(email_to_remove)
        new_admin_str = ",".join(admin_emails)

        # Update .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        update_env_file(env_path, "ADMIN_LIST", new_admin_str)

        # Update environment variable for current process
        os.environ["ADMIN_LIST"] = new_admin_str

        print(f"✅ Removed {email_to_remove} from admin list")

        return jsonify({
            "message": f"Successfully removed {email_to_remove} from admin list",
            "emails": admin_emails
        })
    except Exception as e:
        print(f"❌ ERROR in remove_from_adminlist: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def update_env_file(env_path, key, value):
    """Update or add a key-value pair in the .env file."""
    # Read existing .env file
    lines = []
    key_found = False

    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()

    # Update or add the key
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            key_found = True
        else:
            new_lines.append(line)

    # If key wasn't found, add it
    if not key_found:
        new_lines.append(f"{key}={value}\n")

    # Write back to .env file
    with open(env_path, 'w') as f:
        f.writelines(new_lines)

    print(f"📝 Updated .env file: {key}={value}")

@app.route('/api/upload-temp', methods=['POST'])
def upload_temp():
    """Handle temporary PDF uploads (not saved to ChromaDB)"""
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

        # Split text into chunks (similar to ChromaDB chunks)
        chunk_size = 500
        chunks = []
        for doc in documents:
            for i in range(0, len(doc), chunk_size):
                chunks.append(doc[i:i + chunk_size])

        print(f"📚 Created {len(chunks)} text chunks from temporary uploads.")

        # Store in temporary documents (just text, ChromaDB will handle embedding on query)
        temp_documents['texts'] = chunks
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
    temp_documents['texts'] = []
    temp_documents['is_processing'] = False
    temp_documents['is_ready'] = False
    print("🗑️ Cleared temporary documents")
    return jsonify({'success': True, 'message': 'Temporary documents cleared'})


# =========================================
# 10. Serve Frontend (Catch-All)
# =========================================
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path.startswith('api/'):
        return jsonify({"error": "API endpoint not found"}), 404
        
    try:
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return app.send_static_file(path)
        else:
            index_path = os.path.join(app.static_folder, 'index.html')
            if os.path.exists(index_path):
                return app.send_static_file('index.html')
            else:
                return jsonify({
                    "error": "Frontend not built",
                    "message": "The React frontend build folder was not found."
                }), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================
# 11. Startup
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
    