import requests
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

def insertar(tabla: str, data: dict):
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/{tabla}",
        json=data,
        headers={**HEADERS, "Prefer": "return=representation"}
    )
    return response.json()

def consultar(tabla: str, filtros: dict = {}):
    params = "&".join([f"{k}=eq.{v}" for k, v in filtros.items()])
    url = f"{SUPABASE_URL}/rest/v1/{tabla}?select=*"
    if params:
        url += f"&{params}"
    response = requests.get(url, headers=HEADERS)
    return response.json()