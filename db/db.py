import os
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()
SUPABASE_URL = os.getenv("PUBLIC_SUPABASE_URL")         # ← Replace with your Supabase URL
SUPABASE_KEY = os.getenv("PUBLIC_SUPABASE_KEY")    # ← Replace with your Supabase Key

Supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)