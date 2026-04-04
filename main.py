import datetime
import os
import pickle

import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import serial
import threading
import json
import time
import requests

# Import your routes and models
from routes.farm import router as farm_router
from schemas import FarmCreate, DiseaseResponse, GridInsightsResponse, UploadImagesResponse
# At the top after imports
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# ========================== CONFIG ==========================
app = FastAPI(
    title="AgriSmart AI Backend",
    description="Backend for Smart Crop Monitoring & 3D Farm Visualization",
    version="1.0.0"
)

# CORS Configuration (Update origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                    # Change this to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase Client Initialization


# Make supabase available globally if needed in routes
# You can also pass it via dependency injection later

# ========================== INCLUDE ROUTES ==========================
app.include_router(farm_router, prefix="/api")

# ========================== ROOT ENDPOINT ==========================
@app.get("/")
async def root():
    return {
        "message": "AgriSmart AI Backend is running successfullydwdwwyhjyj 🚀",
        "documentation": "/docs",
        "version": "1.0.0"

    }

# ========================== HEALTH CHECK ==========================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agrismart-ai-backendsxzzxzdssdsdsd"}



app = FastAPI()

ser = serial.Serial("COM6", 9600, timeout=1)
latest_sensor = None

# # ─────────────────────────────
# # 1️⃣ READ SERIAL CONTINUOUSLY
# # ─────────────────────────────
def read_serial_continuously():
    global latest_sensor
    print("Listening to Arduino on COM7...")

    # Create data folder automatically
    import os
    os.makedirs("data", exist_ok=True)

    while True:
        try:
            line = ser.readline().decode().strip()

            if line.startswith("{"):
                latest_sensor = json.loads(line)

                with open("data/sensor.json", "w") as f:
                    json.dump(latest_sensor, f, indent=2)

                print("Arduino:", latest_sensor)

        except Exception as e:
            print("Serial Error:", e)

        time.sleep(0.1)


threading.Thread(target=read_serial_continuously, daemon=True).start()
#
# # ───────────────────────────────────────────────────────────────
# # 2️⃣ UPLOAD LATEST SENSOR DATA EVERY 5 MINUTES TO YOUR API ENDPOINT
# # ───────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/sensor"

def upload_every_5_minutes():
    while True:
        if latest_sensor:
            try:
                print("Uploading to cloud:", latest_sensor)
                response = requests.post(API_URL, json=latest_sensor, timeout=10)
                print("Cloud Response:", response.status_code, response.text)
            except Exception as e:
                print("Upload error:", e)

        time.sleep(30)  # 300 seconds = 5 minutes

threading.Thread(target=upload_every_5_minutes, daemon=True).start()


# ─────────────────────────────
# 3️⃣ FASTAPI ENDPOINTS
# ─────────────────────────────

@app.get("/sensor")
def get_sensor():
    print("sensor data received")
    return {"latest_sensor": latest_sensor}


@app.get("/predict")
def predict():
    if latest_sensor is None:
        return {"error": "No sensor data yet"}

    t = latest_sensor.get("temperature")
    h = latest_sensor.get("humidity")

    prediction = "Normal"
    if t > 30:
        prediction = "High Temperature Warning"

    return {"sensor": latest_sensor, "prediction": prediction}

# --- CORS for frontend / IoT ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Bhuvan WMS Settings ---
WMS_BASE_URL = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
LAYER_MAP = {
    "ndvi": "bhuvan:LULC_250K",
    "vegetation": "lulc:LULC50K_1516",
    "lulc": "lulc:LULC50K_1516",
    "soil": "soil:SOIL_TEXTURE",
    "zinc": "bhuvan:INDIA_STATE",
    "iron": "bhuvan:INDIA_STATE",
    "boundary": "bhuvan:INDIA_STATE",
}

class BboxRequest(BaseModel):
    minLat: float
    minLon: float
    maxLat: float
    maxLon: float
    nutrient: str


# ✅ Load trained model (make sure it's in the same folder)
model = pickle.load(open("xgb_irrigation_model.pkl", "rb"))

# --- WeatherAPI settings ---
WEATHER_API_KEY = "9980c416231943d3ba5132023250412"
FORECAST_URL = "https://api.weatherapi.com/v1/forecast.json"
HISTORY_URL = "https://api.weatherapi.com/v1/history.json"
HTTP_TIMEOUT = 15

latest_sensor_from_device = None


# --- Helper functions ---
def _safe_get_totalprecip_mm(day_obj) -> float:
    try:
        return float(day_obj.get("totalprecip_mm", 0.0))
    except Exception:
        return 0.0


def _history_precip_mm(lat: float, lon: float, date_str: str) -> float:
    params = {
        "key": WEATHER_API_KEY,
        "q": f"{lat},{lon}",
        "dt": date_str,
        "aqi": "no",
        "alerts": "no",
    }
    r = requests.get(HISTORY_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    day = data["forecast"]["forecastday"][0]["day"]
    return _safe_get_totalprecip_mm(day)


def fetch_weather_and_moisture_change(lat: float, lon: float):
    f_params = {
        "key": WEATHER_API_KEY,
        "q": f"{lat},{lon}",
        "days": 4,
        "aqi": "no",
        "alerts": "no",
    }
    f_res = requests.get(FORECAST_URL, params=f_params, timeout=HTTP_TIMEOUT)
    f_res.raise_for_status()
    f_json = f_res.json()

    forecast_days = f_json["forecast"]["forecastday"]
    rainfall_mm_today = _safe_get_totalprecip_mm(forecast_days[0]["day"])
    next3 = forecast_days[1:4] if len(forecast_days) > 1 else []
    rainfall_forecast_next_3days_mm = float(sum(_safe_get_totalprecip_mm(d["day"]) for d in next3))

    today = datetime.utcnow().date()
    past_dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in (1, 2, 3)]
    past_vals = []
    for ds in past_dates:
        try:
            past_vals.append(_history_precip_mm(lat, lon, ds))
        except Exception:
            pass

    past3_avg = float(np.mean(past_vals)) if past_vals else rainfall_mm_today
    delta = rainfall_mm_today - past3_avg
    scale = max(5.0, 0.5 * (rainfall_mm_today + past3_avg))
    soil_moisture_change_percent = 20.0 * float(np.tanh(delta / scale))

    return {
        "rainfall_mm_today": float(rainfall_mm_today),
        "rainfall_forecast_next_3days_mm": float(rainfall_forecast_next_3days_mm),
        "soil_moisture_change_percent": float(soil_moisture_change_percent),
    }


# --- Root ---
@app.get("/")
def home():
    return {
        "message": "🌾 Smart Irrigation API — Auto Prediction Enabled",
        "endpoints": {
            "/sensor (POST)": "Send sensor data — prediction triggers automatically",
            "/sensor (GET)": "Get latest sensor data + last prediction",
            "/predict": "Manual prediction (if needed)",
            "/weather": "Fetch weather data",
        },
    }


# --- Weather ---
@app.get("/weather")
def get_weather(latitude: float = 14.45, longitude: float = 75.90):
    weather = fetch_weather_and_moisture_change(latitude, longitude)
    return weather


# --- Save sensor data + auto prediction ---
@app.post("/sensor")
def receive_sensor_data(sensor: dict):
    """
    Receives sensor data and instantly runs irrigation prediction.
    """
    global latest_sensor_from_device
    required_keys = ["temperature", "humidity", "soil_moisture_raw", "soil_moisture_percent"]

    if not all(k in sensor for k in required_keys):
        raise HTTPException(status_code=400, detail="Missing required sensor fields")

    latest_sensor_from_device = {
        "temperature": float(sensor["temperature"]),
        "humidity": float(sensor["humidity"]),
        "soil_moisture_raw": float(sensor["soil_moisture_raw"]),
        "soil_moisture_percent": float(sensor["soil_moisture_percent"]),
        "timestamp": datetime.now().isoformat(),
    }

    # Save to JSON
    with open("latest_sensor.json", "w") as f:
        json.dump(latest_sensor_from_device, f, indent=4)

    # --- Auto trigger prediction ---
    try:
        result = auto_predict_irrigation(latest_sensor_from_device)
        latest_sensor_from_device["predicted_irrigation_mm_day"] = result["predicted_irrigation_mm_day"]
        latest_sensor_from_device["prediction_message"] = result["message"]
    except Exception as e:
        latest_sensor_from_device["prediction_message"] = f"Prediction failed: {e}"

    # Save updated result
    with open("latest_sensor.json", "w") as f:
        json.dump(latest_sensor_from_device, f, indent=4)

    return {"status": "ok", "data": latest_sensor_from_device}


# --- Get latest sensor data ---
@app.get("/sensor")
def get_latest_sensor_data():
    global latest_sensor_from_device
    if latest_sensor_from_device is None and os.path.exists("latest_sensor.json"):
        with open("latest_sensor.json", "r") as f:
            latest_sensor_from_device = json.load(f)

    if latest_sensor_from_device is None:
        return {"status": "no_data", "error": "No sensor data received yet"}
    print(latest_sensor_from_device)
    return {"status": "ok", "data": latest_sensor_from_device}


# --- Helper: Auto prediction when new sensor arrives ---
def auto_predict_irrigation(sensor_data):
    latitude, longitude = 14.45, 75.90  # fixed for Davangere
    soil_moisture_percent = sensor_data.get("soil_moisture_percent", 45.0)
    soil_temperature_c = sensor_data.get("temperature", 28.0)

    weather = fetch_weather_and_moisture_change(latitude, longitude)
    features = np.array([[
        latitude,
        longitude,
        soil_moisture_percent,
        soil_temperature_c,
        weather["soil_moisture_change_percent"],
        weather["rainfall_mm_today"],
        weather["rainfall_forecast_next_3days_mm"],
    ]])

    prediction = float(model.predict(features)[0])
    prediction = max(0.0, prediction)

    # Log to CSV
    with open("prediction_logs.csv", "a") as f:
        f.write(
            f"{datetime.now()},{latitude},{longitude},{soil_moisture_percent},"
            f"{soil_temperature_c},{weather['rainfall_mm_today']},"
            f"{weather['rainfall_forecast_next_3days_mm']},{prediction}\n"
        )
    with open("moisture_data.csv","a") as f1:
        f1.write(f"{soil_moisture_percent}")

    result = {
        "predicted_irrigation_mm_day": round(prediction, 2),
        "message": "Auto prediction successful ✅",
        "soil_moisture_percent":soil_moisture_percent  ,
        "soil_temperature_c" :soil_temperature_c
    }
    print(result)
    return result


# --- Manual prediction endpoint (optional) ---
# @app.post("/predict")
# def manual_predict(
#     latitude: float = Query(14.45),
#     longitude: float = Query(75.90),
#     soil_moisture_percent: float | None = Query(None),
#     soil_temperature_c: float | None = Query(None),
# ):
#     # Read latest sensor if missing
#     if (soil_moisture_percent is None or soil_temperature_c is None) and os.path.exists("latest_sensor.json"):
#         with open("latest_sensor.json", "r") as f:
#             sensor_data = json.load(f)
#         soil_moisture_percent = soil_moisture_percent or sensor_data.get("soil_moisture_percent", 45.0)
#         soil_temperature_c = soil_temperature_c or sensor_data.get("temperature", 28.0)
#     return auto_predict_irrigation({
#         "soil_moisture_percent": soil_moisture_percent,
#         "temperature": soil_temperature_c
#     })


# ========================== RUN SERVER ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True          # Set to False in production
    )