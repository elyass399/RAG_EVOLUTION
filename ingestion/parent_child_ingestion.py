import os, json, asyncio, re, logging, time, subprocess
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "parentchild"
MODEL_EMB = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# --- BRIDGE AZIENDALE ---
def _ps_invoke_sync(url_suffix, payload):
    url = f"{os.getenv('LITELLM_BASE_URL').rstrip('/')}/{url_suffix}"
    api_key = os.getenv("LITELLM_API_KEY")
    stamp = f"{int(time.time() * 1000)}_{os.getpid()}"
    req_f = Path(f"req_pc_{stamp}.json").absolute().as_posix()
    res_f = Path(f"res_pc_{stamp}.bin").absolute().as_posix()
    try:
        with open(req_f, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False)
        ps_cmd = f'$u="{url}"; $h=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json"}}; $b=[System.IO.File]::ReadAllBytes("{req_f}"); try{{$r=Invoke-WebRequest -Uri $u -Method Post -Headers $h -Body $b -Proxy $null -UseBasicParsing -TimeoutSec 120; [System.IO.File]::WriteAllBytes("{res_f}", $r.RawContentStream.ToArray())}} catch{{}}'
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, check=False)
        if os.path.exists(res_f):
            with open(res_f, "rb") as f: return json.loads(f.read().decode("utf-8-sig"))
    finally:
        if os.path.exists(req_f): os.remove(req_f)
        if os.path.exists(res_f): os.remove(res_f)
    return None

async def ai_call(url_suffix, payload):
    return await asyncio.to_thread(_ps_invoke_sync, url_suffix, payload)

# --- LOGICA CHUNKING GERARCHICO ---

def extract_parents(text):
    """Crea i blocchi grandi (Parent)."""
    # Prova split strutturale
    pattern = r'(?=\d{1,2}\)\s-)|(?=\n#+\s)'
    parents = [p.strip() for p in re.split(pattern, text) if len(p.strip()) > 30]
    
    # Se non trova nulla con la regex, usa il fallback per paragrafi
    if len(parents) <= 1:
        parents = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
    return parents

def extract_children(parent_text):
    """Divide il genitore in frasi (Child)."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', parent_text) if len(s.strip()) > 15]

# --- PROCESSO ---

async def process_manual(manual_path, semaphore):
    logger.info(f"📖 Lettura file: {manual_path.name}")
    text = manual_path.read_text(encoding="utf-8")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    parents = extract_parents(text)
    logger.info(f"   ∟ {manual_path.name}: Identificati {len(parents)} blocchi PADRE.")

    all_points = []
    point_count = 0

    for p_idx, p_text in enumerate(parents):
        children = extract_children(p_text)
        
        for c_idx, c_text in enumerate(children):
            async with semaphore:
                res = await ai_call("embeddings", {"model": MODEL_EMB, "input": c_text})
                if res:
                    point_count += 1
                    all_points.append(PointStruct(
                        id=hash(f"{manual_path.stem}_{p_idx}_{c_idx}_{time.time()}") & 0xFFFFFFFFFFFFFFFF,
                        vector=res['data'][0]['embedding'],
                        payload={
                            "text": c_text,
                            "parent_text": p_text,
                            "source": manual_path.name
                        }
                    ))

    if all_points:
        client.upsert(collection_name=COLLECTION, points=all_points)
        logger.info(f"✅ {manual_path.name}: Caricati {point_count} punti FIGLIO.")
    return point_count

async def run():
    # Rilevamento cartella root
    cur = Path(__file__).parent.absolute()
    base_dir = cur.parent if cur.name == "notebooks" else cur
    input_dir = base_dir / "manuals"
    
    print(f"🔍 [DEBUG] Cerco manuali in: {input_dir}")
    
    if not input_dir.exists():
        print(f"❌ ERRORE: La cartella {input_dir} non esiste!")
        return

    manuals = list(input_dir.glob("*.md"))
    if not manuals:
        print(f"❌ ERRORE: Nessun file .md trovato in {input_dir}")
        return

    if not client.collection_exists(COLLECTION):
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=768, distance=Distance.COSINE))
    
    semaphore = asyncio.Semaphore(5)
    tasks = [process_manual(m, semaphore) for m in manuals]
    results = await asyncio.gather(*tasks)
    
    print(f"\n🏆 INGESTIONE COMPLETATA! Totale punti caricati: {sum(results)}")

if __name__ == "__main__":
    asyncio.run(run())