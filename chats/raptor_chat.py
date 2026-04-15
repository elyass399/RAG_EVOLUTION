import os, json, requests, numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from rank_bm25 import BM25Okapi

# --- 1. CONFIG ---
load_dotenv()
# Costruzione proxy sicura per tutte le versioni di Python
user = os.getenv('PROXY_USER', '').replace('@', '%40')
password = os.getenv('PROXY_PASS', '')
host = os.getenv('PROXY_HOST', '')
proxy_string = f"http://{user}:{password}@{host}"
PROXIES = {"http": proxy_string, "https": proxy_string}

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "zucchetti_raptor_kb"
BASE_URL = os.getenv("LITELLM_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("LITELLM_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
client = QdrantClient(url=QDRANT_URL)

# --- 2. COMUNICAZIONE ROBUSTA ---
def call_api(endpoint, payload, timeout=60):
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, proxies=PROXIES, timeout=timeout, verify=False)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"DEBUG: Errore API {endpoint} ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"DEBUG: Errore connessione {endpoint}: {e}")
    return None

def call_llm(context, question):
    payload = {
        "model": os.getenv("MODEL_SUMMARY", "gemma4:26b"),
        "messages": [
            {"role": "system", "content": "Sei l'assistente Zucchetti. Rispondi usando solo il contesto fornito."},
            {"role": "user", "content": f"CONTESTO:\n{context}\n\nDOMANDA: {question}"}
        ],
        "temperature": 0.1
    }
    data = call_api("chat/completions", payload, timeout=600)
    return data['choices'][0]['message']['content'] if data and 'choices' in data else "Errore: Modello non ha risposto."

# --- 3. LOGICA RAG ---
def search_top_l0(query_vec, query_text):
    # Recupero L0
    docs = client.scroll(COLLECTION_NAME, scroll_filter=Filter(must=[FieldCondition(key="level", match=MatchValue(value=0))]), limit=500, with_payload=True)[0]
    if not docs: return []
    
    # BM25 (Ranking)
    bm25 = BM25Okapi([d.payload.get('text', '').lower().split() for d in docs])
    scores = bm25.get_scores(query_text.lower().split())
    
    combined = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for s, d in combined[:3]]

def get_parents(l0_nodes):
    p_ids = {n.payload.get(k) for n in l0_nodes for k in ['L1ID', 'L2ID'] if n.payload.get(k)}
    if not p_ids: return []
    return client.scroll(COLLECTION_NAME, scroll_filter=Filter(must=[FieldCondition(key="node_id", match=MatchAny(any=list(p_ids)))]), limit=10, with_payload=True)[0]

# --- 4. CHAT LOOP ---
def start_chat():
    print("\n🚀 ZUCCHETTI RAG RAPTOR - Ready")
    while True:
        q = input("\n👤 Tu: ").strip()
        if q.lower() in ["esci", "quit"]: break
        if not q: continue
        
        # Chiamata Embedding con controllo errori
        emb_data = call_api("embeddings", {"model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"), "input": q}, timeout=30)
        if not emb_data or 'data' not in emb_data:
            print("❌ Errore: Il server non ha restituito l'embedding. Verifica il proxy.")
            continue
            
        vec = emb_data['data'][0]['embedding']
        l0 = search_top_l0(vec, q)
        
        if l0:
            nodes = sorted(l0 + get_parents(l0), key=lambda x: x.payload.get('level', 0), reverse=True)
            context = "\n\n".join([f"---[ Livello {n.payload.get('level')} ]---\n{n.payload.get('text')}" for n in nodes[:3]])
            
            print("   📝 Generazione risposta...")
            print(f"\n🤖 Zucchetti AI:\n{call_llm(context, q)}")
        else:
            print("\n🤖 Nessuna info trovata.")

if __name__ == "__main__":
    start_chat()