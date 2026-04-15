import os, subprocess, json, time, re
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import numpy as np

# --- 1. SETUP AMBIENTE ---
load_dotenv()
# Escludiamo il proxy per parlare con Docker locale e il server interno di Padova
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

# --- 2. CONFIGURAZIONE ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "zucchetti_knowledge_base"
VECTOR_SIZE = 768 # Nomic-embed-text standard
CHUNKS_DIR = Path("output_final_zucchetti")

client = QdrantClient(url=QDRANT_URL)

# --- 3. BRIDGE POWERSHELL (Versione Robusta con ID unico) ---
def _ps_invoke(url_suffix, payload, unique_id):
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL").rstrip('/')
    url = f"{base_url}/{url_suffix}"
    
    # Nome file unico per evitare sovrapposizioni nelle chiamate veloci
    req_file = Path(f"ireq_{unique_id}.json").absolute().as_posix()
    res_file = Path(f"ires_{unique_id}.bin").absolute().as_posix()

    with open(req_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    ps_cmd = f'$url="{url}"; $headers=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json;charset=utf-8"}}; try{{$body=[System.IO.File]::ReadAllBytes("{req_file}");$res=Invoke-WebRequest -Uri $url -Method Post -Headers $headers -Body $body -Proxy $null -UseBasicParsing -TimeoutSec 60;[System.IO.File]::WriteAllBytes("{res_file}", $res.RawContentStream.ToArray())}} catch{{$errMsg="ERROR: "+$_.Exception.Message;[System.IO.File]::WriteAllBytes("{res_file}", [System.Text.Encoding]::UTF8.GetBytes($errMsg))}}'
    
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    
    data = None
    if os.path.exists(res_file):
        with open(res_file, "rb") as f: raw = f.read()
        if not raw.startswith(b"ERROR"):
            try: data = json.loads(raw.decode('utf-8-sig'))
            except: data = None
    
    # Pulizia
    if os.path.exists(req_file): os.remove(req_file)
    if os.path.exists(res_file): os.remove(res_file)
    return data

def get_embedding(text, unique_id):
    """Calcola il vettore solo per il testo pulito"""
    data = _ps_invoke("embeddings", {"model": os.getenv("EMBEDDING_MODEL"), "input": text}, unique_id)
    if data and 'data' in data:
        return np.array(data['data'][0]['embedding']).flatten().tolist()
    return None

# --- 4. MOTORE DI INGESTION ---
def run_clean_ingestion():
    print(f"🧨 Reset database: eliminazione collezione '{COLLECTION_NAME}'...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except: pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("✅ Database pronto.\n")

    global_id = 1
    # Scansioniamo le cartelle calcio, guerre, prompt
    manual_folders = [d for d in CHUNKS_DIR.iterdir() if d.is_dir()]
    
    for manual_folder in manual_folders:
        print(f"📦 Caricamento manuale: {manual_folder.name.upper()}")
        files = sorted(list(manual_folder.glob("*.md")))
        
        for file_path in files:
            raw_text = file_path.read_text(encoding="utf-8")
            
            # --- LOGICA DI SEPARAZIONE ---
            # Cerchiamo solo la prima occorrenza di [CONTESTO: ...] all'inizio
            pattern = r'^\[CONTESTO:\s*(.*?)\]'
            match = re.search(pattern, raw_text, re.DOTALL)
            
            if match:
                context_header = match.group(1).strip()
                pure_text = raw_text[match.end():].strip()
            else:
                context_header = "N/D"
                pure_text = raw_text

            # --- GENERAZIONE VETTORE (SOLO TESTO PURO) ---
            vector = get_embedding(pure_text, global_id)
            
            if vector:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id=global_id,
                        vector=vector,
                        payload={
                            "text": pure_text,       # Per il Reranking
                            "context": context_header, # Per la coerenza di Gemma
                            "manual": manual_folder.name,
                            "filename": file_path.name
                        }
                    )]
                )
                print(f"   [{global_id}] ✅ {file_path.name} indicizzato.")
                global_id += 1
            else:
                print(f"   [{global_id}] ❌ ERRORE: {file_path.name}")
            
            # Piccola pausa per non intasare il server
            time.sleep(0.02)

    print(f"\n🚀 FINE. Totale punti caricati: {global_id - 1}")

if __name__ == "__main__":
    run_clean_ingestion()