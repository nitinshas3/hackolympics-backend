import os
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()
SUPABASE_URL = "https://tuozfrhjymolovdzwsff.supabase.co"       # ← Replace with your Supabase URL
SUPABASE_KEY = "sb_publishable_aNpXY-I9LHo-pirvigfOaw_QWNoEq_8"   # ← Replace with your Supabase Key

Supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)