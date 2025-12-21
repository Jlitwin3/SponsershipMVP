import os
import dropbox
import fitz  # PyMuPDF
from io import BytesIO
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.expanduser("~/.chroma_db_data"))

if not DROPBOX_ACCESS_TOKEN:
    print("❌ Dropbox token missing in .env")
if not OPENROUTER_API_KEY:
    print("❌ OpenRouter API key missing in .env")

# Initialize Dropbox Client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN) if DROPBOX_ACCESS_TOKEN else None
DROPBOX_FOLDER = "/L’mu-Oa (Sports Sponsorship AI Project)"

# Initialize Embedding Function (Must match dbAccess2.py)
openrouter_embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENROUTER_API_KEY,
    model_name="text-embedding-3-small",
    api_base="https://openrouter.ai/api/v1"
)

def fetch_dropbox_files(extensions, folder_path=DROPBOX_FOLDER):
    """Get list of files with specific extensions in Dropbox folder (recursively)."""
    if not dbx: return []
    print(f"📂 Scanning Dropbox folder for {extensions}: {folder_path}")
    files = []
    try:
        result = dbx.files_list_folder(folder_path, recursive=True)
        while True:
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata) and any(entry.name.lower().endswith(ext) for ext in extensions):
                    files.append(entry)
            if not result.has_more: break
            result = dbx.files_list_folder_continue(result.cursor)
    except Exception as e:
        print(f"❌ Dropbox error: {e}")
    return files

def extract_text_from_pdf(file_metadata):
    _, res = dbx.files_download(file_metadata.path_lower)
    pdf_bytes = BytesIO(res.content)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def index_pdfs_if_new(collection, processing_status=None):
    if not dbx: return
    if processing_status: processing_status["is_processing"] = True
    
    pdf_files = fetch_dropbox_files((".pdf"))
    print(f"Found {len(pdf_files)} PDFs. Checking for updates...")
    
    for file_meta in pdf_files:
        existing = collection.get(where={"source": file_meta.name})
        if existing["metadatas"] and existing["metadatas"][0].get("rev") == file_meta.rev:
            print(f"⏭️  Skipping {file_meta.name} (unchanged)")
            continue

        print(f"📄 Processing PDF: {file_meta.name}")
        try:
            text = extract_text_from_pdf(file_meta)
            if not text.strip(): continue
            
            size = 1000
            chunks = [text[i:i+size] for i in range(0, len(text), size)]
            
            base_id = file_meta.id or file_meta.name
            ids = [f"{base_id}_chunk{i}" for i in range(len(chunks))]
            
            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=[{"source": file_meta.name, "rev": file_meta.rev, "type": "pdf"}] * len(chunks)
            )
            print(f"✅ Indexed {len(chunks)} chunks.")
        except Exception as e:
            print(f"❌ Error processing {file_meta.name}: {e}")

def index_images_if_new(collection, processing_status=None):
    if not dbx or not collection: return
    
    image_files = fetch_dropbox_files((".jpg", ".jpeg", ".png"))
    print(f"Found {len(image_files)} images. Checking for updates...")
    
    for file_meta in image_files:
        existing = collection.get(where={"source": file_meta.name})
        if existing["metadatas"] and existing["metadatas"][0].get("rev") == file_meta.rev:
            print(f"⏭️  Skipping {file_meta.name} (unchanged)")
            continue

        print(f"🖼️  Processing Image: {file_meta.name}")
        try:
            # For now, we index the filename and path as context
            # In the future, this can be expanded with OCR
            context = f"Image file: {file_meta.name}\nPath: {file_meta.path_display}"
            
            collection.add(
                ids=[file_meta.id or file_meta.name],
                documents=[context],
                metadatas=[{"source": file_meta.name, "rev": file_meta.rev, "type": "image"}]
            )
            print(f"✅ Indexed image metadata.")
        except Exception as e:
            print(f"❌ Error processing {file_meta.name}: {e}")

    if processing_status:
        processing_status["is_processing"] = False
        processing_status["is_ready"] = True
    print("✨ Sync Complete!")

if __name__ == "__main__":
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # PDF Collection
    pdf_col = client.get_or_create_collection(
        name="pdf_embeddings", 
        embedding_function=openrouter_embed
    )
    index_pdfs_if_new(pdf_col)
    
    # Image Collection
    img_col = client.get_or_create_collection(
        name="image_embeddings",
        embedding_function=openrouter_embed
    )
    index_images_if_new(img_col)
