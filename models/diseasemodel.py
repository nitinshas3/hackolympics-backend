import google.generativeai as genai
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

# ========================== GEMINI SETUP ==========================
GEMINI_API_KEY ="AIzaSyDJGRhUBvhP3sZ-mluaCtJ-NPWP3TCewm0"

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is not set in .env file!")
    gemini_model = None
else:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini model initialized successfully")


def predict_disease(image_bytes: bytes, grid_pos: str = "", crop_type: str = "Crop") -> dict:
    """Get disease + comprehensive remedy from Gemini"""
    if gemini_model is None:
        return {
            "disease": "Unknown",
            "confidence": 0.0,
            "is_healthy": True,
            "remedy": "Gemini API key not configured",
            "note": "API key missing"
        }

    prompt = f"""
    You are an expert agriculturist for small farmers in Karnataka, India.

    Analyze this leaf image from grid {grid_pos} of {crop_type} crop.

    Return **ONLY valid JSON** (no extra text) in this exact format:

    {{
      "disease": "Healthy" or "Specific Disease Name (e.g. Early Blight, Bacterial Spot, Leaf Rust)",
      "confidence": number between 0 and 100,
      "is_healthy": true or false,
      "remedy": "Write in simple English. Include:
         - Immediate action (prefer organic first)
         - Exact dosage / quantity (e.g. 2g per liter water)
         - How to prepare and apply
         - When and how often to apply
         - Preventive tips for next season"
    }}

    Use simple, practical language. Mention common local names if helpful.
    """

    try:
        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_bytes
                        }
                    }
                ]
            }]
        )

        text = response.text.strip()

        # Extract JSON safely
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(text)

        return {
            "disease": result.get("disease", "Unknown"),
            "confidence": float(result.get("confidence", 50)),
            "is_healthy": result.get("is_healthy", False),
            "remedy": result.get("remedy", "Consult local agriculture officer.")
        }

    except Exception as e:
        print(f"Gemini vision error: {e}")
        return {
            "disease": "Unknown",
            "confidence": 30.0,
            "is_healthy": True,
            "remedy": "Unable to generate remedy right now. Please try again or consult local expert.",
            "note": "API error"
        }


# ==================== TEST BLOCK ====================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    ASSETS_DIR = BASE_DIR / "assets"

    test_images = ["00.jpg", "01.jpg", "02.jpg", "03.jpg", "09.jpg", "11.jpg"]

    print(f"Looking in folder: {ASSETS_DIR}\n")

    for img_name in test_images:
        test_image_path = ASSETS_DIR / img_name
        if test_image_path.exists():
            print(f"\n{'='*80}")
            print(f"Testing image: {img_name}")
            with open(test_image_path, "rb") as f:
                image_bytes = f.read()

            result = predict_disease(image_bytes, grid_pos="(2, 1)", crop_type="Tomato")

            print("🔍 Final Result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))   # ← This prevents Unicode escaping
            print("-" * 80)
        else:
            print(f"❌ Image not found: {img_name}")

    print("\nTest finished. Check if remedy now shows proper English with dosage.")