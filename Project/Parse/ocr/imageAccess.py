import os
import numpy as np
from io import BytesIO
from PIL import Image
import dropbox
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import threading

try:
    import pytesseract
    OCR_LIBRARY = "tesseract"
    print("📝 Using Tesseract OCR")
except ImportError:
    try:
        import easyocr
        OCR_LIBRARY = "easyocr"
        print("📝 Using EasyOCR")
    except ImportError:
        raise ImportError("Please install either easyocr or pytesseract for OCR functionality")

# =========================================
# 0. Configuration
# =========================================
SKIP_DROPBOX_INDEXING = True  # Set to True to skip Dropbox scanning (we want to scan images!)

# =========================================
# 1. Load environment variables
# =========================================
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
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
DROPBOX_FOLDER = ""  # Scan entire Dropbox recursively for images

# =========================================
# 3. ChromaDB Setup (Persistent) - SAME DB, SEPARATE COLLECTION
# =========================================
# Use the same ChromaDB path as databaseAccess.py
chroma_db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=chroma_db_path)

# Try to get existing collection first, create if it doesn't exist
try:
    image_collection = client.get_collection("image_embeddings")
    print(f"✅ Image collection found. Current count: {image_collection.count()}")
except:
    # Collection doesn't exist, create it
    try:
        image_collection = client.create_collection(
            name="image_embeddings",
            metadata={"description": "OCR text from images"}
        )
        print(f"✅ Created new image collection")
    except Exception as e:
        print(f"❌ Error with image collection: {e}")
        raise

# =========================================
# 4. Flask Setup
# =========================================
app = Flask(__name__)
CORS(app)

processing_status = {"is_processing": False, "is_ready": True}

# =========================================
# 5. OCR Initialization
# =========================================
if OCR_LIBRARY == "easyocr":
    # Initialize EasyOCR reader (supports multiple languages)
    try:
        reader = easyocr.Reader(['en'])  # Add more languages as needed: ['en', 'es', 'fr']
        print("✅ Using EasyOCR for text extraction")
    except Exception as e:
        print(f"⚠️  EasyOCR initialization failed: {e}")
        print("⚠️  Falling back to Tesseract")
        OCR_LIBRARY = "tesseract"
elif OCR_LIBRARY == "tesseract":
    print("✅ Using Tesseract OCR for text extraction")

# =========================================
# 6. Helper Functions
# =========================================
def fetch_dropbox_images(folder_path=DROPBOX_FOLDER):
    """Get list of image files (JPG, JPEG, PNG) in Dropbox folder (recursively)."""
    print(f"📂 Scanning Dropbox folder: {folder_path}")

    image_files = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    try:
        result = dbx.files_list_folder(folder_path, recursive=True)
    except dropbox.exceptions.ApiError as e:
        print("❌ Dropbox API error:", e)
        return image_files

    # Process first batch
    for entry in result.entries:
        if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith(supported_extensions):
            image_files.append(entry)

    # Continue paging if there are more results
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith(supported_extensions):
                image_files.append(entry)

    print(f"📸 Found {len(image_files)} image(s).")
    return image_files


def extract_text_from_image(image_data):
    """Extract text from image using OCR."""
    try:
        # Open image from bytes
        image = Image.open(BytesIO(image_data))

        # Convert to RGB if necessary (some formats like PNG with transparency)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Perform OCR
        if OCR_LIBRARY == "easyocr":
            # EasyOCR returns list of (bbox, text, confidence)
            result = reader.readtext(np.array(image))
            text = " ".join([detection[1] for detection in result])
        elif OCR_LIBRARY == "tesseract":
            # Convert PIL Image to numpy array for Tesseract
            text = pytesseract.image_to_string(np.array(image))

        return text.strip()

    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return ""


def extract_text_from_dropbox_image(file_metadata):
    """Download and extract text from a Dropbox image."""
    try:
        _, res = dbx.files_download(file_metadata.path_lower)
        image_bytes = res.content
        text = extract_text_from_image(image_bytes)
        return text
    except Exception as e:
        print(f"❌ Error downloading/processing {file_metadata.name}: {e}")
        return ""


def chunk_text(text, size=500):
    """Split text into smaller chunks."""
    if not text:
        return []
    return [text[i:i+size] for i in range(0, len(text), size)]


# =========================================
# 7. Embedding Function
# =========================================
openrouter_embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name="text-embedding-3-large"
)

# =========================================
# 8. Index Images Once
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


def index_images_if_new():
    """Index only new or modified images from Dropbox."""
    processing_status["is_processing"] = True
    image_files = fetch_dropbox_images()

    existing_ids = get_all_ids(image_collection)
    print(f"📊 Existing image embeddings: {len(existing_ids)}")

    for file_metadata in image_files:
        image_id = file_metadata.id or file_metadata.name

        # Check if this file already exists in DB
        existing = image_collection.get(where={"source": file_metadata.name})
        if existing["metadatas"]:
            existing_rev = existing["metadatas"][0].get("rev")
            # Skip if same revision (unchanged file)
            if existing_rev == file_metadata.rev:
                print(f"⏭️  Skipping {file_metadata.name} — no changes detected.")
                continue

        # Skip if already indexed
        if image_id in existing_ids:
            print(f"⏭️  Already indexed: {file_metadata.name}")
            continue

        # Otherwise, process and index
        print(f"🖼️  Processing: {file_metadata.name}")
        try:
            text = extract_text_from_dropbox_image(file_metadata)

            if not text:
                print(f"⚠️  Skipping {file_metadata.name} — no text extracted from OCR.")
                continue

            chunks = chunk_text(text)

            if not chunks:
                print(f"⚠️  Skipping {file_metadata.name} — no chunks created.")
                continue

            base_id = image_id if image_id else file_metadata.name.replace(" ", "_").replace(".", "_")
            ids = [f"{base_id}_chunk{i}" for i in range(len(chunks))]

            # Include revision info in metadata
            image_collection.add(
                ids=ids,
                documents=chunks,
                metadatas=[{
                    "source": file_metadata.name,
                    "rev": file_metadata.rev,
                    "type": "image"
                }] * len(chunks)
            )

            print(f"✅ Indexed {len(chunks)} chunks from {file_metadata.name}")

        except Exception as e:
            print(f"❌ Failed to process {file_metadata.name}: {e}")

    processing_status["is_processing"] = False
    processing_status["is_ready"] = True
    print("📚 All Dropbox images processed and stored in ChromaDB!")


# =========================================
# 9. Status Endpoint
# =========================================
@app.route("/api/image-status", methods=["GET"])
def image_status():
    """Get status of image collection."""
    return jsonify({
        "processed": processing_status["is_ready"],
        "is_processing": processing_status["is_processing"],
        "collection_size": image_collection.count(),
        "ocr_library": OCR_LIBRARY
    })


# =========================================
# 10. Startup
# =========================================
if __name__ == "__main__":
    if not SKIP_DROPBOX_INDEXING:
        thread = threading.Thread(target=index_images_if_new)
        thread.daemon = True
        thread.start()
        print("🔄 Dropbox indexing enabled. Images will be scanned and updated.")
    else:
        processing_status["is_ready"] = True
        print("⏭️  Skipping Dropbox indexing — using existing ChromaDB data only.")

    print("\n🚀 Flask server running at http://localhost:5002")
    app.run(debug=True, port=5002, use_reloader=False)
