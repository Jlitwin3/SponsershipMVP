# Deployment Guide for Sponsor Scout

This application is ready for deployment on **Render** (recommended) or AWS Lightsail.

## 1. Prerequisites
*   **GitHub Repository**: Ensure this code is pushed to your GitHub repo.
*   **Render Account**: Create one at [render.com](https://render.com).
*   **API Keys**: You will need the following keys:
    *   `GOOGLE_API_KEY` (Gemini AI)
    *   `DROPBOX_ACCESS_TOKEN` (For PDF indexing)
    *   `RAPIDAPI_KEY` (For LinkedIn data)
    *   `OPENROUTER_API_KEY` (Optional, if using other models)

## 2. Prepare Frontend (Important!)
Since Render's Python environment doesn't build React apps by default, the easiest way is to commit your local build:
1.  Open `.gitignore` in the root folder.
2.  Remove or comment out the line: `Webapp/frontend/build/`
3.  Run these commands locally:
    ```bash
    cd Webapp/frontend
    npm run build
    cd ../..
    git add Webapp/frontend/build
    git commit -m "Add frontend build for deployment"
    git push origin main
    ```

## 3. Deploying to Render
1.  **New Web Service**: Connect your GitHub repo.
2.  **Runtime**: Python 3.
3.  **Build Command**: `pip install -r Project/Parse/requirements.txt`
4.  **Start Command**: `gunicorn --timeout 300 --workers 2 --bind 0.0.0.0:$PORT Project.Parse.databaseAccess:app`
    *   *Note: The `Procfile` in `Project/Parse/Procfile` also defines this.*
5.  **Root Directory**: `Project/Parse` (Important!)

## 3. Environment Variables
Go to the **Environment** tab in Render and add:
*   `PYTHON_VERSION`: `3.11.0`
*   `TOKENIZERS_PARALLELISM`: `false`
*   `GOOGLE_API_KEY`: [Your Key]
*   `DROPBOX_ACCESS_TOKEN`: [Your Key]
*   `RAPIDAPI_KEY`: [Your Key]
*   `SKIP_DROPBOX_INDEXING`: `False` (Set to `True` if you don't want to re-scan Dropbox on every restart)

## 4. Persistent Storage (Optional but Recommended)
To keep your PDF index from disappearing on every restart:
1.  Go to **Disks** in Render.
2.  Add a disk:
    *   **Mount Path**: `/opt/render/.chroma_db_data`
    *   **Size**: 1 GB
3.  Add Env Var: `CHROMA_DB_PATH=/opt/render/.chroma_db_data`

## 5. Troubleshooting
*   **Timeout Errors**: We increased the timeout to 300s (5 mins) to allow the AI model to load.
*   **Worker Killed**: If you see "Worker was sent SIGKILL", it usually means the free tier memory (512MB) was exceeded. Upgrade to the Starter plan ($7/mo) for 2GB RAM.
