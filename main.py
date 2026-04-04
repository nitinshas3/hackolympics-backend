import datetime
import json
import os
import threading
import time
import requests
import numpy as np
import pickle

import serial
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ========================== IMPORTS ==========================
from routes.farm import router as farm_router

# ========================== CONFIG ==========================
app = FastAPI(
    title="AgriSmart AI Backend",
    description="Backend for Smart Crop Monitoring, 3D Farm Visualization & Precision Irrigation",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================== SUPABASE ==========================
SUPABASE_URL = os.getenv("PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("PUBLIC_SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# ========================== SENSOR & IRRIGATION SETUP ==========================
ser = None
latest_sensor = None
latest_irrigation_prediction = None

# Arduino Connection
try:
    ser = serial.Serial("COM6", 9600, timeout=1)
    print("✅ Arduino connected on COM6")
except Exception as e:
    print(f"⚠️ Arduino not connected: {e}. Running without hardware.")

# Load Irrigation Model
try:
    model = pickle.load(open("xgb_irrigation_model.pkl", "rb"))
    print("✅ Irrigation XGBoost model loaded successfully")
except Exception as e:
    print(f"⚠️ Irrigation model failed to load: {e}")
    model = None

# ========================== WEATHER HELPER FUNCTION (Fixed) ==========================
def fetch_weather_and_moisture_change(lat: float, lon: float):
    """Your original weather function - simplified for stability"""
    try:
        WEATHER_API_KEY = "016f934d566c45dcb57163906260404"   # Put in .env later
        FORECAST_URL = "https://api.weatherapi.com/v1/forecast.json"

        params = {
            "key": WEATHER_API_KEY,
            "q": f"{lat},{lon}",
            "days": 4,
            "aqi": "no",
            "alerts": "no"
        }
        r = requests.get(FORECAST_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        forecast_days = data["forecast"]["forecastday"]
        rainfall_today = float(forecast_days[0]["day"].get("totalprecip_mm", 0.0))
        rainfall_next3 = sum(float(d["day"].get("totalprecip_mm", 0.0)) for d in forecast_days[1:4])

        return {
            "rainfall_mm_today": rainfall_today,
            "rainfall_forecast_next_3days_mm": rainfall_next3,
            "soil_moisture_change_percent": round(15.0 + (rainfall_today * 2), 2)
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return {
            "rainfall_mm_today": 0.0,
            "rainfall_forecast_next_3days_mm": 5.0,
            "soil_moisture_change_percent": 12.0
        }


# ========================== AUTO IRRIGATION PREDICTION ==========================
def auto_predict_irrigation(sensor_data: dict):
    global latest_irrigation_prediction
    if model is None:
        return {"error": "Irrigation model not loaded"}

    try:
        lat, lon = 12.2958, 76.6394  # Mysuru

        weather = fetch_weather_and_moisture_change(lat, lon)

        features = np.array([[
            lat,
            lon,
            float(sensor_data.get("soil_moisture_percent", 45.0)),
            float(sensor_data.get("temperature", 28.0)),
            weather["soil_moisture_change_percent"],
            weather["rainfall_mm_today"],
            weather["rainfall_forecast_next_3days_mm"],
        ]])

        prediction = float(model.predict(features)[0])
        prediction = max(0.0, round(prediction, 2))

        latest_irrigation_prediction = {
            "predicted_irrigation_mm_day": prediction,
            "message": "Irrigation recommendation updated successfully",
            "soil_moisture_percent": sensor_data.get("soil_moisture_percent"),
            "temperature": sensor_data.get("temperature"),
            "timestamp": datetime.datetime.now().isoformat()
        }

        return latest_irrigation_prediction

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": f"Prediction failed: {str(e)}"}


# ========================== SERIAL READING ==========================
def read_serial_continuously():
    global latest_sensor
    print("🎧 Listening to Arduino...")

    os.makedirs("data", exist_ok=True)

    while True:
        try:
            if ser and ser.is_open:
                line = ser.readline().decode('utf-8').strip()
                if line and line.startswith("{"):
                    latest_sensor = json.loads(line)
                    with open("data/sensor.json", "w") as f:
                        json.dump(latest_sensor, f, indent=2)
                    print(f"📡 Arduino: {latest_sensor}")
        except:
            pass
        time.sleep(0.2)


if ser:
    threading.Thread(target=read_serial_continuously, daemon=True).start()


# ========================== FASTAPI ROUTES ==========================
app.include_router(farm_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "AgriSmart AI Backend is running successfully 🚀",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Sensor Routes
@app.get("/sensor")
async def get_sensor():
    global latest_sensor
    if latest_sensor is None and os.path.exists("data/sensor.json"):
        with open("data/sensor.json", "r") as f:
            latest_sensor = json.load(f)
    return {"latest_sensor": latest_sensor or "No sensor data yet"}


@app.post("/sensor")
async def receive_sensor_data(sensor: dict):
    global latest_sensor
    latest_sensor = sensor

    with open("data/sensor.json", "w") as f:
        json.dump(sensor, f, indent=2)

    # Auto trigger irrigation prediction
    prediction = auto_predict_irrigation(sensor)

    return {
        "status": "success",
        "sensor_data": sensor,
        "irrigation_prediction": prediction
    }


@app.get("/predict")
async def get_irrigation_prediction():
    global latest_irrigation_prediction
    if latest_irrigation_prediction is None:
        return {"error": "No prediction available yet. Send sensor data first."}
    return latest_irrigation_prediction


# ========================== RUN SERVER ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False      # Set to True only during development
    )