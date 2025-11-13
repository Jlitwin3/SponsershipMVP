# Parse Folder - Project Structure

## Overview
Backend services for L'mu-Oa sports sponsorship AI assistant.

## Folder Structure

```
Parse/
├── databaseAccess.py          # Main chat API server (port 5001)
├── .env                        # API keys and configuration
├── chroma_db/                  # Vector database (PDFs + Images)
│
├── sponsorship/                # 🆕 Sponsor database & query intelligence
│   ├── __init__.py
│   ├── sponsor_manager.py      # Database operations (CRUD, conflict checks)
│   ├── database_schema.py      # SQLite schema definition
│   ├── sponsors.db             # Sponsor database
│   ├── populate_uo_sponsors.py # Script to populate real UO sponsor data
│   └── query_classifier.py     # Intelligent query categorization
│
├── ocr/                        # 🆕 Image processing
│   ├── __init__.py
│   └── imageAccess.py          # OCR server (port 5002)
│
└── utils/                      # 🆕 Utility scripts
    ├── __init__.py
    ├── OpenRouter.py           # OpenRouter API utilities
    ├── addNewFile.py           # Add new PDFs to database
    ├── deleteRepeatEmbeddings.py
    ├── listCollections.py      # View ChromaDB collections
    └── read_from_DB.py         # Query database directly
```

## Main Services

### 1. databaseAccess.py (Port 5001)
**Main chat server** - handles all user queries

**Features:**
- Query classification (sponsor checks, temporal queries, etc.)
- Sponsor database integration
- Vector search (PDFs + images)
- AI response generation via OpenRouter

**Endpoints:**
- `POST /api/chat` - Main chat endpoint
- `GET /api/status` - Server status
- `GET /api/sponsors` - List all sponsors
- `GET /api/sponsors/<name>` - Get sponsor details

**Start:**
```bash
python3 databaseAccess.py
```

### 2. ocr/imageAccess.py (Port 5002)
**Image processing server** - OCR and indexing

**Features:**
- Scans Dropbox for images
- OCR text extraction (Tesseract)
- Stores embeddings in ChromaDB

**Endpoint:**
- `GET /api/image-status` - Processing status

**Start:**
```bash
python3 ocr/imageAccess.py
```

## Sponsorship Module

### sponsor_manager.py
Core database operations for sponsor management.

**Key Functions:**
- `check_sponsor_conflict(name, category)` - Check if sponsor conflicts
- `get_sponsor_info(name)` - Get detailed sponsor info
- `get_all_current_sponsors()` - List active sponsors
- `add_sponsor(...)` - Add new sponsor to database

**Example:**
```python
from sponsorship.sponsor_manager import check_sponsor_conflict

result = check_sponsor_conflict("Nike")
if result['conflict']:
    print(f"Conflict: {result['details']}")
```

### query_classifier.py
Intelligent query categorization system.

**Query Types:**
- `SPONSOR_CHECK` - "Can we propose Nike?"
- `TEMPORAL` - "What sponsors signed recently?"
- `HISTORICAL` - "Tell me about our 1990s sponsors"
- `ANALYSIS` - "Compare our sponsors"
- `LIST_REQUEST` - "Who are our Charter Partners?"
- `GENERAL` - Everything else

**Example:**
```python
from sponsorship.query_classifier import QueryClassifier

classifier = QueryClassifier()
result = classifier.classify("Can we propose Nike?")
# {'type': 'sponsor_check', 'needs_db': True, 'entities': ['Nike']}
```

### database_schema.py
SQLite database schema definition.

**Run to create/reset database:**
```bash
python3 sponsorship/database_schema.py
```

### populate_uo_sponsors.py
Populates database with real University of Oregon sponsors.

**Run once to load data:**
```bash
python3 sponsorship/populate_uo_sponsors.py
```

**Sponsors included:**
- Nike (exclusive athletic apparel)
- Bigfoot Beverages (exclusive pouring rights)
- 14 other current sponsors
- Historical data and partnerships

## Development Workflow

### First Time Setup
```bash
# 1. Create database
python3 sponsorship/database_schema.py

# 2. Populate with UO sponsors
python3 sponsorship/populate_uo_sponsors.py

# 3. Start servers
python3 databaseAccess.py          # Terminal 1
python3 ocr/imageAccess.py         # Terminal 2
cd ../../Webapp/frontend && npm start  # Terminal 3
```

### Adding New Sponsors
```python
from sponsorship.sponsor_manager import add_sponsor

add_sponsor(
    name="Company Name",
    category="Technology",
    relationship_type="current",
    is_exclusive=False,
    start_date="2024-01-01",
    notes="Partnership details..."
)
```

### Testing Queries
```python
from sponsorship.query_classifier import QueryClassifier

classifier = QueryClassifier()
result = classifier.classify("Your query here")
print(result)
```

## Configuration

### .env File
Required environment variables:
```
OPENROUTER_API_KEY=your_api_key
DROPBOX_ACCESS_TOKEN=your_dropbox_token
```

### databaseAccess.py Settings
```python
SKIP_DROPBOX_INDEXING = True  # Skip PDF re-indexing on startup
```

### ocr/imageAccess.py Settings
```python
SKIP_DROPBOX_INDEXING = True  # Skip image re-indexing on startup
OCR_LIBRARY = "tesseract"     # or "easyocr"
```

## Utils Scripts

### View Collections
```bash
python3 utils/listCollections.py
```

### Query Database Directly
```bash
python3 utils/read_from_DB.py
```

### Add New PDF
```bash
python3 utils/addNewFile.py path/to/file.pdf
```

## Notes

- The sponsor database (`sponsorship/sponsors.db`) contains real UO partnership data
- ChromaDB (`chroma_db/`) stores embeddings for PDFs and images
- Both servers must run simultaneously for full functionality
- Query classification happens automatically in databaseAccess.py
