#!/bin/bash

# RAG Chat Backend Startup Script

echo "🚀 Starting RAG Chat Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if [ ! -f "venv/lib/python*/site-packages/flask/__init__.py" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your OPENROUTER_API_KEY"
    exit 1
fi

# Start the Flask app
echo "✅ Starting Flask server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
python app.py
