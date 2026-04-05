# AgriSmart AI - Smart Farm Monitoring Backend

A FastAPI-based backend for precision agriculture. It helps farmers monitor crops, detect plant diseases grid-wise, generate 3D farm insights, and receive practical remedial recommendations (including dosage and application methods).

## Features

- Farm creation and management
- Upload overview images for farm mapping
- Grid-based leaf disease detection using AI
- Comprehensive remedy suggestions with dosage and application instructions
- Grid-wise insights storage (disease, severity, fertilizer, irrigation, pesticide, remedy)
- Supabase integration (Database + Storage)

## Tech Stack

- **Framework**: FastAPI
- **Database**: Supabase (PostgreSQL)
- **AI Model**: Gemini 1.5 Flash (Vision) for disease detection + remedy
- **Image Handling**: Async file uploads
- **Environment Management**: python-dotenv

## Project Structure
finalbackend/
├── assets/                  # Test leaf images
├── data/                    # sensor.json etc.
├── db/
│   └── db.py                # Supabase client
├── models/
│   └── diseasemodel.py      # Gemini Vision prediction + remedy
├── routes/
│   └── farm.py              # All API routes
├── schemas/
│   └── init.py          # Pydantic models
├── .env                     # Environment variables
├── main.py                  # FastAPI app entry point
├── requirements.txt
└── README.md

## Setup Instructions

### 1. Clone / Setup Project

```bash
git clone <your-repo>
cd finalbackend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate