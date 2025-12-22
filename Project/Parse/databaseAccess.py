import os
# Disable tokenizers parallelism to prevent deadlocks in Gunicorn
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import fitz
from io import BytesIO
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
import time
import psutil

# =========================================
# 1. Load environment variables
# =========================================
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.expanduser("~/.chroma_db_data"))

if not OPENROUTER_API_KEY:
    print("⚠️  Warning: OPENROUTER_API_KEY not found in .env")
if not GOOGLE_API_KEY:
    print("⚠️  Warning: GOOGLE_API_KEY not found in .env")

# Add current directory to path for local modules
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

# =========================================
# 2. Embedding Functions
# =========================================
class OpenRouterEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key, model_name="text-embedding-3-small"):
        self.api_key = api_key
        self.model_name = model_name
        self.url = "https://openrouter.ai/api/v1/embeddings"

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "input": input
            }
        )
        response.raise_for_status()
        data = response.json()
        # OpenRouter returns a list of objects with an "embedding" field
        return [item["embedding"] for item in data["data"]]

# This is the "key" to seeing your data
openrouter_embed = OpenRouterEmbeddingFunction(
    api_key=OPENROUTER_API_KEY,
    model_name="text-embedding-3-small"
)

# =========================================
# 3. ChromaDB Setup
# =========================================
if not os.path.exists(CHROMA_DB_PATH):
    os.makedirs(CHROMA_DB_PATH)

print(f"📂 ChromaDB path: {CHROMA_DB_PATH}", flush=True)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# We pass the embedding_function here so ChromaDB can decode the data
def get_safe_collection(name, metadata=None):
    try:
        # Try to get with the desired embedding function
        return client.get_collection(name=name, embedding_function=openrouter_embed)
    except ValueError as e:
        if "Embedding function conflict" in str(e):
            print(f"⚠️  Embedding function conflict for {name}. Checking if empty...")
            # Get without embedding function to check count
            temp_col = client.get_collection(name=name)
            if temp_col.count() == 0:
                print(f"🗑️  Collection {name} is empty but has wrong config. Resetting...")
                client.delete_collection(name=name)
                return client.create_collection(name=name, embedding_function=openrouter_embed, metadata=metadata)
            else:
                print(f"❌  Collection {name} contains data with a different embedding function!")
                raise e
        else:
            # If it doesn't exist, create it
            return client.create_collection(name=name, embedding_function=openrouter_embed, metadata=metadata)
    except Exception:
        # General fallback to create
        return client.get_or_create_collection(name=name, embedding_function=openrouter_embed, metadata=metadata)

collection = get_safe_collection("pdf_embeddings", {"description": "PDF research papers"})
print(f"✅ PDF collection loaded. Count: {collection.count()}", flush=True)

image_collection = get_safe_collection("image_embeddings")
if image_collection:
    print(f"✅ Image collection loaded. Count: {image_collection.count()}", flush=True)

# =========================================
# 4. Initialize Gemini
# =========================================
print("🤖 Initializing Gemini...", flush=True)
if GOOGLE_API_KEY:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
else:
    gemini_client = None
    grounding_tool = None

# =========================================
# 5. Flask Setup
# =========================================
print("Initializing Build Folder")
BUILD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Webapp', 'frontend', 'build')
print("Build Folder Initialized")
print("Initializing Flask App")
app = Flask(__name__, static_folder=BUILD_FOLDER, static_url_path='/')
print("Flask App Initialized")
CORS(app)
print("CORS started")

print("Initializing Upload Folder")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
print("Upload Folder Initialized")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print("NO UPLOAD FOLDER: Upload Folder Created")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print("Upload Folder Configured")

print("Initializing Processing Status")
processing_status = {"is_processing": False, "is_ready": True}
print("Processing Status Initialized")
chat_history = []
print("Chat History Initialized")
temp_documents = {'texts': [], 'is_processing': False, 'is_ready': False}
print("Temp Documents Initialized")
query_classifier = QueryClassifier()
print("Query Classifier Initialized")

# =========================================
# 6. Helper Functions
# =========================================
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return f"{mem_mb:.2f} MB"

def extract_sponsor_mentions(query: str) -> list:
    print("Extracting Sponsor Mentions")
    sponsor_keywords = ["propose", "sponsor", "partnership", "deal", "agreement"]
    query_lower = query.lower()
    if not any(kw in query_lower for kw in sponsor_keywords):
        return []
    words = query.split()
    potential_sponsors = []
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)
        if clean_word and clean_word[0].isupper() and clean_word.lower() not in ['can', 'we', 'should', 'would', 'what', 'how', 'why', 'when', 'where']:
            potential_sponsors.append(clean_word)
    return potential_sponsors

# =========================================
# 7. Routes
# =========================================

@app.route("/api/status", methods=["GET"])
def status():
    try:
        # Get counts safely
        pdf_count = 0
        img_count = 0
        try:
            pdf_count = collection.count()
            if image_collection:
                img_count = image_collection.count()
        except Exception as e:
            print(f"⚠️  Error counting collections: {e}")

        return jsonify({
            "processed": True, # Always true now since we skip indexing
            "is_processing": False,
            "collection_size": pdf_count,
            "image_collection_size": img_count,
            "total_documents": pdf_count + img_count
        })
    except Exception as e:
        print(f"❌ Status route critical error: {e}")
        return jsonify({
            "processed": True,
            "is_processing": False,
            "error": str(e)
        })

@app.route("/api/chat", methods=["POST"])
def chat():
    start_time = time.time()
    try:
        print(f"📊 [MEM] Start: {get_memory_usage()}", flush=True)
        print("getting data json")
        data = request.get_json(silent=True)
        print("data json received")
        if not data or not data.get("query"):
            print("No query provided")
            return jsonify({"error": "No query provided"}), 400
        
        user_query = data.get("query")
        print(f"🚀 [DEBUG] Received chat query: {user_query}", flush=True)

        # 1. Classify
        print("Classifying query")
        classification = query_classifier.classify(user_query)
        print("Query classified")
        
        # 2. Sponsor DB Context
        print("Getting sponsor context")
        sponsor_context = ""
        sponsor_mentions = classification['entities'] if classification['needs_db'] else extract_sponsor_mentions(user_query)
        print("sponsor mentions gathered")
        if sponsor_mentions:
            for sponsor_name in sponsor_mentions:
                sponsor_info = get_sponsor_info(sponsor_name)
                if sponsor_info:
                    sponsor_context += format_sponsor_context(sponsor_info) + "\n"
                
                conflict_info = check_sponsor_conflict(sponsor_name)
                if conflict_info['conflict']:
                    sponsor_context += f"\n[Important Sponsorship Notice]\nRegarding {sponsor_name}:\n{conflict_info['details']}\n"

        # 3. Generate Query Embedding (Once)
        print(f"🧠 [DEBUG] Generating query embedding... (Mem: {get_memory_usage()})", flush=True)
        embed_start = time.time()
        query_embedding = openrouter_embed([user_query])[0]
        print(f"✅ [DEBUG] Embedding generated in {time.time() - embed_start:.2f}s (Mem: {get_memory_usage()})", flush=True)

        # 4. Vector Search
        if classification['type'] == query_classifier.TEMPORAL:
            pdf_count, image_count = 7, 4
        elif classification['type'] == query_classifier.LIST_REQUEST:
            pdf_count, image_count = 8, 4
        else:
            pdf_count, image_count = 5, 3

        print(f"🔍 [DEBUG] Querying PDF collection...", flush=True)
        search_start = time.time()
        pdf_results = collection.query(query_embeddings=[query_embedding], n_results=pdf_count)
        print("pdf results gathered")
        all_documents = pdf_results["documents"][0] if pdf_results["documents"] else []
        all_metadatas = pdf_results["metadatas"][0] if pdf_results["metadatas"] else []
        print("all documents and metadatas gathered")

        if image_collection:
            try:
                print(f"🔍 [DEBUG] Querying Image collection...", flush=True)
                img_results = image_collection.query(query_embeddings=[query_embedding], n_results=image_count)
                print("image results gathered")
                if img_results["documents"]:
                    all_documents.extend(img_results["documents"][0])
                    all_metadatas.extend(img_results["metadatas"][0])
            except Exception as e:
                print(f"⚠️ Image query failed: {e}")
        print(f"✅ [DEBUG] Vector search complete in {time.time() - search_start:.2f}s (Mem: {get_memory_usage()})", flush=True)

        # 5. Build Context
        context_parts = []
        print("building context")
        if sponsor_context: context_parts.append(sponsor_context)
        print("sponsor context added")
        for doc, meta in zip(all_documents, all_metadatas):
            source_type = meta.get('type', 'pdf')
            source_name = meta.get('source', 'unknown')
            context_parts.append(f"[From {source_type}: {source_name}]\n{doc}")
            print("document added to context")
        
        if temp_documents['is_ready']:
            context_parts.extend([f"[From temporary upload]\n{t}" for t in temp_documents['texts'][:5]])

        context = "\n\n".join(context_parts)
        
        # 5. LLM Call
        if not gemini_client:
            print("Gemini client not initialized")
            return jsonify({"error": "Gemini API key not configured"}), 500
        print("Gemini client initialized")
        
        system_prompt = """You are the "Sponsor Scout" AI Agent, an expert research assistant for students seeking sponsorships for their class project or event. Your primary goal is to provide accurate, actionable, and relevant information regarding potential sponsors, their giving criteria, contact information, and deadlines.
        🎯 Core Persona & Goal
        Role: Expert research assistant and database navigator.

        you must use google if you cannot find the info in the documents
        YOU MUST USE GOOGLE IF YOU CANNOT FIND THE INFO IN THE DOCUMENTS

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

        5. **Use LinkedIn Tool for Social Media Updates**:
           - You have access to a tool called `fetch_linkedin_updates` (via the "Fresh LinkedIn Profile Data" API).
           - Use this tool AUTOMATICALLY when the user asks for "recent posts", "updates", "social media", or "LinkedIn" activity.
           - The tool requires a LinkedIn Company URL. If you don't have it, try to infer it or ask the user, but usually you can proceed if the tool is enabled.
           - CRITICAL: NEVER say "I don't have access to live LinkedIn data" or "I cannot browse social media". You HAVE this tool. Use it.


        📝 Constraints and Guidelines
        Filter Irrelevant Content: Ignore general news or marketing material. Focus strictly on corporate giving, philanthropy, and sponsorship programs.

        Handle Ambiguity: If a student's request is too vague (e.g., "Who will sponsor us?"), ask clarifying questions such as, "What is your project/event focused on (e.g., technology, environment, arts)?" and "What is your target funding amount?"

        No Personal Contact: Do not generate or provide personal contact details (e.g., direct email addresses of employees). Only provide links to official application forms or general corporate contact pages.

        Maintain Data Security: Do not reveal the structure or filenames of the internal database; only cite the company or document name.

        🚀 Example Interaction
        User: "I'm looking for information on corporate sponsorships for a high school robotics team. Do we have any info on Microsoft?"

        Your Expected Response: (Check database first, then use your knowledge)

        "That's a great project! Microsoft is a strong fit.
        
        According to the 'Microsoft Giving Guidelines 2024' PDF, they prioritize programs focused on closing the digital skills gap and supporting underrepresented groups in STEM. Applications for grants under $5,000 must be submitted by October 1st annually.
        
        Additionally, their AI for Good program sometimes extends to student projects that utilize emerging technology.

        Actionable Summary: Focus your application on the digital skills gap and ensure you apply before the October 1st deadline for smaller grants.

        What other companies would you like me to look up?"
        """
        
        # Manage chat history
        if len(chat_history) > 10:
            chat_history.pop(0)
        chat_history.append({"role": "user", "content": user_query})

        full_prompt = f"{system_prompt}\n\nChat History:\n"
        for msg in chat_history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"
            
        full_prompt += f"\nContext:\n{context}\n\nUser Question: {user_query}"

        print(f"🚀 [DEBUG] Calling Gemini...", flush=True)
        llm_start = time.time()
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(tools=[grounding_tool])
        )
        print(f"✅ [DEBUG] Gemini responded in {time.time() - llm_start:.2f}s (Mem: {get_memory_usage()})", flush=True)
        
        chat_history.append({"role": "assistant", "content": response.text})
        print(f"✨ [DEBUG] Total chat time: {time.time() - start_time:.2f}s (Final Mem: {get_memory_usage()})", flush=True)
        return jsonify({"answer": response.text})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- Admin Routes ---

@app.route("/api/admin/verify-email", methods=["POST"])
def verify_email():
    from user_db import is_admin
    data = request.json
    email = data.get("email", "").strip().lower()
    return jsonify({"authorized": is_admin(email)})

@app.route("/api/chatbot/verify-email", methods=["POST"])
def verify_chatbot_email():
    from user_db import is_authorized_for_chatbot
    data = request.json
    email = data.get("email", "").strip().lower()
    return jsonify({"authorized": is_authorized_for_chatbot(email)})

@app.route("/api/admin/files", methods=["GET"])
def get_admin_files():
    files_dict = {}
    
    # 1. Fetch from PDF collection
    pdf_results = collection.get(include=["metadatas"])
    for metadata in pdf_results["metadatas"]:
        name = metadata.get("source")
        if name and name not in files_dict:
            files_dict[name] = {
                "name": name, 
                "type": metadata.get("type", "PDF").upper(), 
                "date": metadata.get("upload_date", "")[:10],
                "size": metadata.get("size", "Unknown")
            }
            
    # 2. Fetch from Image collection
    img_results = image_collection.get(include=["metadatas"])
    for metadata in img_results["metadatas"]:
        name = metadata.get("source")
        if name and name not in files_dict:
            files_dict[name] = {
                "name": name, 
                "type": metadata.get("type", "IMAGE").upper(), 
                "date": metadata.get("upload_date", "")[:10],
                "size": metadata.get("size", "Unknown")
            }
            
    return jsonify({"files": list(files_dict.values())})

@app.route("/api/admin/reset_db", methods=["POST"])
def reset_db():
    try:
        client.delete_collection("pdf_embeddings")
        client.delete_collection("image_embeddings")
        global collection, image_collection
        collection = client.create_collection(name="pdf_embeddings", embedding_function=openrouter_embed)
        image_collection = client.create_collection(name="image_embeddings", embedding_function=openrouter_embed)
        
        return jsonify({"message": "Database collections reset. Please run dropbox_indexer.py manually to re-index."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/delete", methods=["POST"])
def delete_admin_file():
    data = request.json
    filename = data.get("filename", "").strip()
    confirm_filename = data.get("confirmFilename", "").strip()

    if not filename or filename != confirm_filename:
        return jsonify({"error": "Filenames do not match or are empty"}), 400

    try:
        # 1. Delete from PDF Collection
        pdf_results = collection.get(where={"source": filename})
        pdf_deleted = 0
        if pdf_results["ids"]:
            collection.delete(ids=pdf_results["ids"])
            pdf_deleted = len(pdf_results["ids"])

        # 2. Delete from Image Collection
        img_results = image_collection.get(where={"source": filename})
        img_deleted = 0
        if img_results["ids"]:
            image_collection.delete(ids=img_results["ids"])
            img_deleted = len(img_results["ids"])

        if pdf_deleted > 0 or img_deleted > 0:
            msg = f"Successfully deleted {filename}. Removed {pdf_deleted} PDF chunks and {img_deleted} image chunks."
            print(f"🗑️ {msg}")
            return jsonify({"success": True, "message": msg})
        else:
            return jsonify({"error": f"File {filename} not found in database"}), 404
    except Exception as e:
        print(f"❌ Error deleting {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/upload", methods=["POST"])
def admin_upload():
    # Support both 'files' (multiple) and 'file' (single) keys
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        if 'file' in request.files:
            uploaded_files = [request.files['file']]
        else:
            print("❌ No file part in request")
            return jsonify({"error": "No file part"}), 400
    
    print(f"📂 Received {len(uploaded_files)} file(s)")
    
    success_count = 0
    errors = []
    
    for file in uploaded_files:
        if file.filename == '':
            continue

        filename = secure_filename(file.filename)
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            print(f"📄 Processing PDF: {filename}")
            try:
                # 1. Extract Text
                pdf_bytes = file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = "".join(page.get_text() for page in doc)
                doc.close()

                if not text.strip():
                    errors.append(f"{filename}: No text found in PDF")
                    continue

                # 2. Chunk Text
                size = 1000
                chunks = [text[i:i+size] for i in range(0, len(text), size)]
                
                # 3. Generate IDs and Metadata
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                file_size_mb = f"{len(pdf_bytes) / (1024 * 1024):.2f} MB"
                base_id = f"admin_{int(time.time())}_{filename}"
                ids = [f"{base_id}_chunk{i}" for i in range(len(chunks))]
                
                metadatas = [{
                    "source": filename,
                    "type": "pdf",
                    "upload_type": "admin_upload",
                    "upload_date": timestamp,
                    "size": file_size_mb
                }] * len(chunks)

                # 4. Add to ChromaDB
                collection.add(
                    ids=ids,
                    documents=chunks,
                    metadatas=metadatas
                )
                success_count += 1
                print(f"✅ Indexed {len(chunks)} chunks from {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                errors.append(f"{filename}: {str(e)}")
        
        elif file_ext in ['jpg', 'jpeg', 'png']:
            print(f"🖼️ Processing Image: {filename}")
            try:
                img_bytes = file.read()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                file_size_mb = f"{len(img_bytes) / (1024 * 1024):.2f} MB"
                
                # Context for image (consistent with dropbox_indexer.py)
                context = f"Image file: {filename}\nUpload Type: Admin Upload\nDate: {timestamp}"
                
                image_collection.add(
                    ids=[f"admin_{int(time.time())}_{filename}"],
                    documents=[context],
                    metadatas=[{
                        "source": filename,
                        "type": "image",
                        "upload_type": "admin_upload",
                        "upload_date": timestamp,
                        "size": file_size_mb
                    }]
                )
                success_count += 1
                print(f"✅ Indexed image metadata for {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                errors.append(f"{filename}: {str(e)}")
        else:
            errors.append(f"{filename}: Invalid file type (PDF, JPG, JPEG, PNG only)")

    if success_count > 0:
        return jsonify({
            "success": True, 
            "message": f"Successfully indexed {success_count} file(s).",
            "errors": errors if errors else None
        })
    else:
        return jsonify({"error": "No files were successfully indexed", "details": errors}), 400

@app.route("/api/admin/whitelist", methods=["GET"])
def get_whitelist():
    from user_db import get_all_whitelisted_users
    return jsonify({"emails": get_all_whitelisted_users()})

@app.route("/api/admin/whitelist/add", methods=["POST"])
def add_to_whitelist():
    from user_db import add_whitelisted_user
    email = request.json.get("email", "").strip().lower()
    if add_whitelisted_user(email):
        return jsonify({"success": True})
    return jsonify({"error": "Already exists"}), 400

@app.route("/api/admin/whitelist/remove", methods=["POST"])
def remove_from_whitelist():
    from user_db import remove_whitelisted_user
    email = request.json.get("email", "").strip().lower()
    if remove_whitelisted_user(email):
        return jsonify({"success": True})
    return jsonify({"error": "Not found"}), 404

@app.route("/api/admin/adminlist", methods=["GET"])
def get_adminlist():
    from user_db import get_all_admins
    return jsonify({"emails": get_all_admins()})

@app.route("/api/admin/adminlist/add", methods=["POST"])
def add_to_adminlist():
    from user_db import add_admin
    email = request.json.get("email", "").strip().lower()
    if add_admin(email):
        return jsonify({"success": True, "message": f"Added {email} as admin"})
    return jsonify({"error": "Already exists"}), 400

@app.route("/api/admin/adminlist/remove", methods=["POST"])
def remove_from_adminlist():
    from user_db import remove_admin
    email = request.json.get("email", "").strip().lower()
    if remove_admin(email):
        return jsonify({"success": True, "message": f"Removed {email} from admins"})
    return jsonify({"error": "Not found"}), 404

# Catch-all for frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path.startswith('api/'): return jsonify({"error": "Not found"}), 404
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return app.send_static_file(path)
    return app.send_static_file('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Server running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
