from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List
import uuid
import json
from models.diseasemodel import predict_disease   # ← Import here
from schemas import (
    FarmCreate,
    DiseaseResponse,
    GridInsightsResponse,
    UploadImagesResponse
)
from db.db import Supabase

router = APIRouter()

# ==================== 1. CREATE FARM ====================
@router.post("/farms")
async def create_farm(farm: FarmCreate):
    try:
        data = farm.model_dump()
        real_user_id = str(uuid.uuid4())
        data["user_id"] = real_user_id

        result = Supabase.table("farms").insert(data).execute()
        farm_id = result.data[0]["id"]

        print(f"✅ Farm created! ID: {farm_id}")
        return {
            "success": True,
            "farm_id": farm_id,
            "user_id": real_user_id,
            "message": "Farm created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 2. UPLOAD OVERVIEW IMAGES (for 3D + Grid) ====================
# Keep your existing upload_farm_images function here (no change needed)

# ==================== GRID-WISE LEAF DISEASE PREDICTION ====================
@router.post("/farms/{farm_id}/upload-leaf-grid")
async def upload_leaf_with_grid(
    farm_id: str,
    files: List[UploadFile] = File(...),
    grid_data: str = Form(...)          # JSON string from frontend
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        grid_info = json.loads(grid_data)
    except:
        raise HTTPException(status_code=400, detail="Invalid grid_data JSON")

    results = []

    for file in files:
        try:
            file_info = next((item for item in grid_info if item.get("filename") == file.filename), None)
            if not file_info:
                continue

            grid_x = int(file_info["grid_x"])
            grid_y = int(file_info["grid_y"])

            image_bytes = await file.read()

            # Upload to Supabase
            file_name = f"{uuid.uuid4()}_{file.filename}"
            Supabase.storage.from_("farm-images").upload(file_name, image_bytes)
            public_url = Supabase.storage.from_("farm-images").get_public_url(file_name)

            Supabase.table("farm_images").insert({
                "farm_id": farm_id,
                "image_url": public_url,
                "image_type": "leaf"
            }).execute()

            # Real AI Prediction
            pred = predict_disease(image_bytes)

            # Save to grid insights
            Supabase.table("farm_grid_insights").insert({
                "farm_id": farm_id,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "disease_detected": not pred["is_healthy"],
                "disease_type": pred["disease"] if not pred["is_healthy"] else None,
                "severity": pred["confidence"] / 100,
                "fertilizer_req": "Apply 18kg NPK" if not pred["is_healthy"] else "Apply 12kg NPK",
                "irrigation_req": "Irrigate 100-120 minutes" if not pred["is_healthy"] else "Irrigate 60 minutes",
                "pesticide_req": "Apply Mancozeb 2g/L" if not pred["is_healthy"] else "No pesticide needed"
            }).execute()

            results.append({
                "filename": file.filename,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "disease": pred["disease"],
                "confidence": pred["confidence"]
            })

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")

    return {
        "message": "Grid-wise disease prediction completed",
        "processed": len(results),
        "results": results
    }
