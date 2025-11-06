# RAG Chat Application

A modern web application that allows you to upload PDF files and chat with them using Retrieval-Augmented Generation (RAG) powered by OpenRouter AI.

## Features

- 📄 **PDF Upload**: Drag-and-drop or select multiple PDF files
- 🤖 **AI Chat**: Ask questions about your uploaded documents
- 🔍 **RAG Pipeline**: Automatically processes documents and retrieves relevant context
- 💬 **Real-time Chat**: Beautiful, responsive chat interface
- ✨ **Modern UI**: Gradient design with smooth animations

## Architecture

- **Backend**: Flask API with PDF processing and RAG implementation
- **Frontend**: React application with modern UI components
- **AI**: OpenRouter API for embeddings and GPT-4o for chat

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- OpenRouter API key

## Quick Start

The fastest way to get started:

1. Set up backend (one-time setup):
```bash
cd Project/Webapp
./start.sh
```

2. In another terminal, set up and start frontend:
```bash
cd Project/Webapp/frontend
npm install
npm start
```

## Detailed Setup Instructions

### 1. Navigate to Webapp Directory

```bash
cd Project/Webapp
```

### 2. Backend Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the `Project/Webapp` directory:

```bash
# Create .env file
cat > Project/Webapp/.env << EOF
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_SITE_URL=http://localhost:3000
OPENROUTER_SITE_TITLE=RAG Chat
EOF
```

**Important**: Replace `your_openrouter_api_key_here` with your actual OpenRouter API key. Get one at [OpenRouter.ai](https://openrouter.ai/).

### 4. Frontend Setup

```bash
cd frontend
npm install
```

### 5. Run the Application

**Terminal 1 - Start the Flask backend:**
```bash
cd Project/Webapp
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

The backend will run on `http://localhost:5000`

**Terminal 2 - Start the React frontend:**
```bash
cd Project/Webapp/frontend
npm start
```

The frontend will automatically open in your browser at `http://localhost:3000`

## Usage

1. **Upload PDFs**: Click "Select Files" or drag-and-drop PDF files
2. **Process**: Click "Process PDFs" to extract text and generate embeddings
3. **Chat**: Ask questions about your documents in the chat interface
4. **Reset**: Click "Reset & Upload New Files" to start over with different documents

## Project Structure

```
Project/
├── Parse/
│   ├── OpenRouter.py      # Original CLI RAG implementation
│   └── asst2.pdf          # Example PDF
├── Webapp/
│   ├── app.py             # Flask backend API
│   ├── requirements.txt   # Python dependencies
│   ├── uploads/           # Temporary PDF storage
│   └── frontend/
│       ├── src/
│       │   ├── App.js     # Main React component
│       │   ├── components/
│       │   │   ├── FileUpload.js      # PDF upload UI
│       │   │   └── ChatInterface.js   # Chat UI
│       │   └── index.js   # Entry point
│       └── package.json   # Node dependencies
└── README.md
```

## API Endpoints

- `POST /api/upload` - Upload and process PDF files
- `POST /api/chat` - Send chat queries and get responses
- `GET /api/status` - Check if documents are processed

## Technologies Used

- **Backend**: Flask, PyMuPDF, scikit-learn, OpenRouter API
- **Frontend**: React, CSS3
- **AI**: GPT-4o, text-embedding-3-large

## Troubleshooting

### Backend won't start
- Ensure your `.env` file has the correct API key
- Check that port 5000 is not in use
- Verify all Python dependencies are installed

### Frontend won't start
- Run `npm install` in the frontend directory
- Check that port 3000 is not in use
- Ensure Node.js version is 16 or higher

### Upload fails
- Check file size (max 16MB)
- Ensure files are valid PDFs
- Check backend console for error messages

## License

MIT License

## Contributing

Feel free to submit issues and enhancement requests!

