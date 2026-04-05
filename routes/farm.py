from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List
import uuid
import json
import os

import google.generativeai as genai

from models.diseasemodel import predict_disease
from schemas import FarmCreate, GridInsightsResponse, UploadImagesResponse
from db.db import Supabase

router = APIRouter()

# ========================== GEMINI SETUP ==========================
GEMINI_API_KEY = "AIzaSyBfsZVzBh8Lqrr69DVuNkzYnHmQ8hhZnYI"

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY is not set in .env!")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini initialized for disease + remedy")


# ==================== 1. CREATE FARM ====================
@router.post("/farms")
async def create_farm(farm: FarmCreate):
    try:
        data = farm.model_dump()
        real_farm_id = f"farm_{uuid.uuid4().hex[:12]}"
        data["farm_id"] = real_farm_id

        result = Supabase.table("farms").insert(data).execute()
        created = result.data[0]

        print(f"✅ Farm created | farm_id: {real_farm_id}")
        return {
            "success": True,
            "farm_id": real_farm_id,
            "name": created.get("name"),
            "message": "Farm created successfully"
        }
    except Exception as e:
        print(f"❌ Farm creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 2. UPLOAD OVERVIEW IMAGES ====================
@router.post("/farms/{farm_id}/upload-images", response_model=UploadImagesResponse)
async def upload_overview_images(
    farm_id: str,
    files: List[UploadFile] = File(...),
    image_type: str = "overview"
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded_count = 0
    for file in files:
        try:
            file_bytes = await file.read()
            file_name = f"{uuid.uuid4()}_{file.filename}"

            Supabase.storage.from_("farm-images").upload(file_name, file_bytes)
            public_url = Supabase.storage.from_("farm-images").get_public_url(file_name)

            Supabase.table("farm_images").insert({
                "farm_id": farm_id,
                "image_url": public_url,
                "image_type": image_type
            }).execute()
            uploaded_count += 1
        except Exception as e:
            print(f"Error uploading overview image: {e}")

    if image_type == "overview" and uploaded_count > 0:
        await analyze_farm_grids(farm_id)

    return UploadImagesResponse(
        message="Overview images uploaded",
        image_count=uploaded_count,
        processed=True
    )


# ==================== 3. BULK GRID-WISE LEAF UPLOAD ====================
@router.post("/farms/{farm_id}/upload-leaf-grid")
async def upload_leaf_with_grid(
    farm_id: str,
    files: List[UploadFile] = File(...),
    grid_data: str = Form(...)
):
    print(f"🚀 Bulk upload started for farm: {farm_id} | Files: {len(files)}")

    if not files:
        raise HTTPException(status_code=400, detail="No image files provided")

    try:
        grid_info = json.loads(grid_data)
        if not isinstance(grid_info, list):
            raise ValueError("grid_data must be a JSON array")
        print(f"Grid data parsed: {len(grid_info)} entries")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid grid_data: {str(e)}")

    results = []
    errors = []

    for file in files:
        try:
            print(f"→ Processing: {file.filename}")

            # Find grid info for this file
            file_info = next((item for item in grid_info if item.get("filename") == file.filename), None)
            if not file_info:
                errors.append(f"No grid info for {file.filename}")
                continue

            grid_x = int(file_info.get("grid_x"))
            grid_y = int(file_info.get("grid_y"))
            grid_pos = f"({grid_x}, {grid_y})"

            image_bytes = await file.read()

            # Call predict_disease with grid info
            try:
                pred = predict_disease(image_bytes, grid_pos=grid_pos, crop_type="Crop")
                print(f"✅ Grid {grid_pos} → {pred.get('disease')} ({pred.get('confidence')}%)")
            except Exception as pred_err:
                print(f"❌ Prediction failed for {file.filename}: {pred_err}")
                pred = {
                    "disease": "Unknown",
                    "confidence": 0.0,
                    "is_healthy": True,
                    "remedy": "Unable to analyze image."
                }

            # Save to Supabase (grid-wise)
            Supabase.table("farm_grid_insights").upsert({
                "farm_id": farm_id,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "disease_detected": not pred.get("is_healthy", True),
                "disease_type": pred.get("disease") if not pred.get("is_healthy", True) else None,
                "severity": pred.get("confidence", 0) / 100.0,
                "fertilizer_req": "Apply 18kg NPK" if not pred.get("is_healthy", True) else "Apply 12kg NPK",
                "irrigation_req": "Irrigate 100-120 minutes" if not pred.get("is_healthy", True) else "Irrigate 60 minutes",
                "pesticide_req": pred.get("remedy", "No pesticide needed") if not pred.get("is_healthy", True) else "No pesticide needed",
                "remedial_measures": pred.get("remedy", "No action needed")
            }).execute()

            print(f"✅ Grid insight saved for {grid_pos}")

            results.append({
                "filename": file.filename,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "disease": pred.get("disease"),
                "confidence": pred.get("confidence", 0),
                "remedy": pred.get("remedy", "")[:250] + "..." if len(pred.get("remedy", "")) > 250 else pred.get("remedy", "")
            })

        except Exception as e:
            error_msg = f"Error with {file.filename}: {str(e)}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)

    print(f"Final Summary - Processed: {len(results)}, Errors: {len(errors)}")
    return {
        "message": "Bulk grid-wise disease detection + remedy completed",
        "processed": len(results),
        "successful": results,
        "errors": errors if errors else None
    }


# ==================== HELPER: CREATE EMPTY GRIDS ====================
async def analyze_farm_grids(farm_id: str):
    try:
        farm = Supabase.table("farms").select("*").eq("farm_id", farm_id).execute().data[0]
        length = farm.get("length_m", 100)
        width = farm.get("width_m", 80)
        grid_size = 10

        grids_x = max(1, int(length // grid_size) + 1)
        grids_y = max(1, int(width // grid_size) + 1)

        Supabase.table("farm_grid_insights").delete().eq("farm_id", farm_id).execute()

        for x in range(grids_x):
            for y in range(grids_y):
                Supabase.table("farm_grid_insights").insert({
                    "farm_id": farm_id,
                    "grid_x": x,
                    "grid_y": y,
                    "disease_detected": False,
                    "severity": 0.0,
                    "fertilizer_req": "Apply 12kg NPK",
                    "irrigation_req": "Irrigate 60 minutes",
                    "pesticide_req": "No pesticide needed",
                    "remedial_measures": "No action needed"
                }).execute()

        print(f"✅ Created {grids_x * grids_y} empty grids for farm {farm_id}")
    except Exception as e:
        print(f"Grid creation error: {e}")


# ==================== GET GRID INSIGHTS ====================
@router.get("/farms/{farm_id}/grids")
async def get_grid_insights(farm_id: str):
    result = Supabase.table("farm_grid_insights")\
        .select("*")\
        .eq("farm_id", farm_id)\
        .execute()

    grids = result.data or []

    if not grids:
        raise HTTPException(status_code=404, detail="No grids found. Upload overview images first.")

    return {
        "farm_id": farm_id,
        "total_grids": len(grids),
        "grid_size_m": 10,
        "grids": grids
    }