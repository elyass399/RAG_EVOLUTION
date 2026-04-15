import os
import requests
from dotenv import load_dotenv, find_dotenv

# 1. Caricamento forza-bruta
env_path = find_dotenv()
load_dotenv(env_path)
print(f"DEBUG: File .env trovato in: {env_path}")

# 2. Configurazione
BASE_URL = os.getenv("LITELLM_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("LITELLM_API_KEY")
user = os.getenv('PROXY_USER', '').replace('@', '%40')
password = os.getenv('PROXY_PASS', '')
host = os.getenv('PROXY_HOST', '')
proxy_string = f"http://{user}:{password}@{host}"
PROXIES = {"http": proxy_string, "https": proxy_string}

print(f"DEBUG: URL target: {BASE_URL}")
print(f"DEBUG: Proxy configurato: {host}")

def test_debug():
    url = f"{BASE_URL}/models"
    headers = {"Authorization": f"Bearer {API_KEY}", "User-Agent": "Mozilla/5.0"}
    
    # TEST A: Connessione forzata senza proxy (per server interno)
    print("\n--- TEST A: Connessione Diretta (Bypass Proxy) ---")
    try:
        r = requests.get(url, headers=headers, proxies={"http": None, "https": None}, timeout=10, verify=False)
        print(f"Risultato: Successo {r.status_code}" if r.status_code == 200 else f"Risultato: Errore {r.status_code}")
    except Exception as e:
        print(f"Errore connessione diretta: {type(e).__name__}: {str(e)[:50]}")

    # TEST B: Connessione tramite Proxy
    print("\n--- TEST B: Connessione tramite Proxy ---")
    try:
        r = requests.get(url, headers=headers, proxies=PROXIES, timeout=10, verify=False)
        print(f"Risultato: Successo {r.status_code}" if r.status_code == 200 else f"Risultato: Errore {r.status_code}")
    except Exception as e:
        print(f"Errore connessione proxy: {type(e).__name__}: {str(e)[:50]}")

if __name__ == "__main__":
    test_debug()