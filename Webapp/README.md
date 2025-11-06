# RAG Chat Web Application

A beautiful web interface for chatting with your PDF documents using Retrieval-Augmented Generation (RAG) powered by OpenRouter AI.

## Features

- 📤 **Upload PDFs**: Drag-and-drop or select multiple PDF files
- 🤖 **AI Chat**: Ask questions and get answers based on your documents
- 🎨 **Modern UI**: Beautiful gradient design with smooth animations
- ⚡ **Fast**: Efficient embedding generation and retrieval
- 💬 **Interactive**: Real-time chat interface with typing indicators

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- OpenRouter API key (get one at https://openrouter.ai/)

## Setup Instructions

### 1. Backend Setup

Navigate to the Webapp directory:
```bash
cd Project/Webapp
```

Create a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

The `.env` file is already configured with your API key. If you need to update it, edit `.env`:
```bash
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_SITE_URL=http://localhost:3000
OPENROUTER_SITE_TITLE=RAG Chat
```

### 2. Frontend Setup

Navigate to the frontend directory:
```bash
cd frontend
```

Install Node dependencies:
```bash
npm install
```

## Running the Application

You need to run both the backend and frontend simultaneously.

### Terminal 1 - Start Backend (Flask)

```bash
cd Project/Webapp
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

The backend will start on `http://localhost:5000`

### Terminal 2 - Start Frontend (React)

```bash
cd Project/Webapp/frontend
npm start
```

The frontend will automatically open in your browser at `http://localhost:3000`

## How to Use

1. **Upload PDFs**:
   - Click "Select Files" or drag-and-drop PDF files onto the upload area
   - Click "Process PDFs" to extract text and generate embeddings
   - Wait for processing to complete (you'll see a loading indicator)

2. **Chat with Your Documents**:
   - Once processing is complete, you'll see the chat interface
   - Type your question in the text box
   - Press Enter or click "Send" to get an AI-generated answer
   - The AI will answer based only on the content of your uploaded PDFs

3. **Upload New Documents**:
   - Click "Reset & Upload New Files" to start over with different PDFs

## API Endpoints

The Flask backend provides these endpoints:

- `POST /api/upload` - Upload and process PDF files
- `POST /api/chat` - Send queries and get AI responses
- `GET /api/status` - Check if documents are processed

## Architecture

### Backend (Flask)
- Handles PDF uploads and text extraction
- Generates embeddings using OpenRouter's `text-embedding-3-large` model
- Performs semantic search to find relevant document chunks
- Uses GPT-4o to generate answers based on retrieved context

### Frontend (React)
- Modern, responsive UI built with React
- File upload with drag-and-drop support
- Real-time chat interface
- Beautiful animations and gradients

## Technology Stack

- **Backend**: Flask, PyMuPDF, scikit-learn, numpy
- **Frontend**: React 18, CSS3
- **AI**: OpenRouter API (GPT-4o + text-embedding-3-large)
- **ML**: Cosine similarity for semantic search

## Troubleshooting

### Backend Issues

**Port 5000 already in use:**
```bash
# Find and kill the process
lsof -ti:5000 | xargs kill -9
```

**API Key errors:**
- Check that your `.env` file has the correct `OPENROUTER_API_KEY`
- Verify the key is valid at https://openrouter.ai/

**PDF extraction fails:**
- Ensure PDFs contain actual text (not just images)
- Try smaller PDFs if you're hitting memory limits

### Frontend Issues

**Port 3000 already in use:**
- The React dev server will offer to use a different port automatically
- Or kill the process: `lsof -ti:3000 | xargs kill -9`

**Network errors:**
- Ensure the Flask backend is running on port 5000
- Check browser console for CORS errors

**Dependencies not installing:**
```bash
# Clear npm cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## File Structure

```
Project/Webapp/
├── app.py                  # Flask backend server
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── uploads/               # Temporary PDF storage
└── frontend/
    ├── package.json       # Node dependencies
    ├── public/
    │   └── index.html
    └── src/
        ├── App.js         # Main React component
        ├── App.css
        ├── index.js
        ├── index.css
        └── components/
            ├── FileUpload.js      # PDF upload component
            ├── FileUpload.css
            ├── ChatInterface.js   # Chat UI component
            └── ChatInterface.css
```

## License

MIT

## Support

For issues or questions, please check:
- OpenRouter API docs: https://openrouter.ai/docs
- Flask docs: https://flask.palletsprojects.com/
- React docs: https://react.dev/
