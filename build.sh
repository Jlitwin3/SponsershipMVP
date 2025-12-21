#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "🔍 Current directory: $(pwd)"
echo "🔍 Directory contents:"
ls -la

# Build Frontend
echo "🚀 Building Frontend..."
cd Webapp/frontend
echo "🔍 Frontend directory: $(pwd)"
npm install
npm run build
echo "🔍 Build completed. Checking build folder..."
ls -la build/ || echo "❌ Build folder not found!"
cd ../..

echo "🔍 Back to root: $(pwd)"
echo "🔍 Checking if build folder exists at Webapp/frontend/build:"
ls -la Webapp/frontend/build/ || echo "❌ Build folder not found at expected location!"

# Install Backend Dependencies
echo "📦 Installing Backend Dependencies..."
pip install -r /Users/jesselitwin/SponsershipMVP/Project/requirements.txt

echo "✅ Build script completed!"
