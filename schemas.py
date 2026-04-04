from pydantic import BaseModel
from typing import List, Optional

class FarmCreate(BaseModel):
    farm_id:str
    name: str
    crop_type: str
    size_acres: float
    length_m: float          # e.g. 100.0
    width_m: float           # e.g. 80.0

class DiseaseResponse(BaseModel):
    detected: bool
    disease_type: Optional[str] = None
    confidence: float
    message: str

class GridInsight(BaseModel):
    grid_x: int
    grid_y: int
    disease_detected: bool
    disease_type: Optional[str]
    severity: float
    fertilizer_req: str
    irrigation_req: str
    pesticide_req: str

class GridInsightsResponse(BaseModel):
    farm_id: str
    total_grids: int
    grid_size_m: int = 10
    grids: List[GridInsight]

class UploadImagesResponse(BaseModel):
    message: str
    image_count: int
    processed: bool