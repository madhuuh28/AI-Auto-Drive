# AI Auto Drive Terrain Classification

FastAPI application that classifies uploaded road-condition images as bad weather, flood water, or mountain/hill terrain and returns a corresponding driving mode.

## Run the application

Install the Python dependencies:

```powershell
python -m pip install fastapi uvicorn python-multipart tensorflow pillow numpy
```

Start the backend:

```powershell
Set-Location "Rain And Snow Images/backend"
python app.py
```

The app listens on `http://localhost:8000`. The frontend is served from `/`, the API health check is at `/health`, and image classification is available at `POST /predict` using a multipart field named `file`.

The model files are kept in the repository. Raw training image collections are excluded by `.gitignore` because they occupy several gigabytes.