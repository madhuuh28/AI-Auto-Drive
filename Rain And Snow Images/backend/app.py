from pathlib import Path
import os

import numpy as np
import tensorflow as tf

from PIL import Image
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "model" / "auto_drive_terrain_model.keras"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


# ============================================
# LOAD AI MODEL
# ============================================

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("AI model loaded successfully! ✅")


# ============================================
# CLASS NAMES
# ============================================

CLASS_NAMES = [
    "Bad_Weather",
    "Flood_Water",
    "Mountain_Hill"
]


# ============================================
# DRIVE MODE SETTINGS
# ============================================

DRIVE_MODES = {
    "Bad_Weather": {
        "mode": "RAIN / SNOW MODE",
        "speed": 40,
        "response": "Reduced throttle + increased stability"
    },

    "Flood_Water": {
        "mode": "WATER MODE",
        "speed": 25,
        "response": "Low speed + controlled torque"
    },

    "Mountain_Hill": {
        "mode": "HILL MODE",
        "speed": 35,
        "response": "Higher torque + controlled acceleration"
    }
}


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="AI Smart Auto Drive System",
    description="AI terrain and weather classification API"
)

app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


# ============================================
# CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# HOME
# ============================================

@app.get("/", response_class=FileResponse)
def home():
    return FRONTEND_DIR / "index.html"


@app.get("/health")
def health():
    return {
        "system": "AI Smart Auto Drive System",
        "status": "Online",
        "message": "AI backend is running successfully!"
    }


# ============================================
# AI PREDICTION
# ============================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])
    vehicle_settings = DRIVE_MODES[predicted_class]

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "drive_mode": vehicle_settings["mode"],
        "recommended_speed": vehicle_settings["speed"],
        "vehicle_response": vehicle_settings["response"]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
