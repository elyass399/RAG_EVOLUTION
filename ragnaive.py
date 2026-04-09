import os, subprocess, json, time, re
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = QdrantClient(url="http://localhost:6333")
COLLECTION = "manuali_late_chunking"

# --- BRIDGE MINIMALE ---
def _ps_invoke(url_suffix, payload):
    api_key = os.getenv("LITELLM_API_KEY")
    url = f"{os.getenv('LITELLM_BASE_URL').rstrip('/')}/{url_suffix}"
    req_file = f"req_{int(time.time())}.json"
    res_file = f"res_{int(time.time())}.bin"
    with open(req_file, "w", encoding="utf-8") as f: json.dump(payload, f)
    
    ps_cmd = f'$u="{url}"; $h=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json"}}; $b=[System.IO.File]::ReadAllBytes("{req_file}"); try{{$r=Invoke-WebRequest -Uri $u -Method Post -Headers $h -Body $b -Proxy $null -UseBasicParsing; [System.IO.File]::WriteAllBytes("{res_file}", $r.RawContentStream.ToArray())}} catch{{}}'
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    
    data = None
    if os.path.exists(res_file):
        with open(res_file, "rb") as f: 
            try: data = json.loads(f.read().decode('utf-8-sig'))
            except: data = None
        os.remove(req_file); os.remove(res_file)
    return data

# --- FUNZIONI AI ---
def get_query_vector(text):
    res = _ps_invoke("embeddings", {"model": os.getenv("EMBEDDING_MODEL"), "input": text})
    return res['data'][0]['embedding'] if res else None

def ask_gemma(context, question):
    prompt = f"Usa questo contesto:\n{context}\n\nRispondi alla domanda: {question}"
    res = _ps_invoke("chat/completions", {"model": "gemma3:27b-it-qat", "messages": [{"role": "user", "content": prompt}]})
    return res['choices'][0]['message']['content'] if res else "Errore"

# --- RERANKER SEMPLICE (Logica Python) ---
def simple_rerank(query, documents):
    """Ri-ordina i documenti contando quante parole della query contengono."""
    query_words = set(query.lower().split())
    scored_docs = []
    
    for doc in documents:
        text = doc.payload.get('text', '').lower()
        # Conta quante parole della domanda ci sono nel testo
        overlap = sum(1 for word in query_words if word in text)
        # Score finale = Score di Qdrant + bonus per parole trovate
        final_score = doc.score + (overlap * 0.1)
        scored_docs.append((final_score, doc))
    
    # Ri-ordina dal più alto al più basso
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [d[1] for d in scored_docs]

# --- CHAT ---
def start_simple_rag():
    print("🚀 RAG Semplice (Retrieval + Rerank manuale)")
    while True:
        query = input("\n👤 Tu: ")
        if query.lower() in ["esci", "quit"]: break

        # 1. Recupero (Top 10 da Qdrant)
        vec = get_query_vector(query)
        results = client.query_points(COLLECTION, query=vec, limit=10).points
        
        if not results:
            print("🤖 Nessun documento trovato."); continue

        # 2. Rerank (Mettiamo in cima quelli con le parole chiave giuste)
        best_results = simple_rerank(query, results)[:3] # Teniamo i top 3 dopo rerank
        
        # 3. Risposta
        context = "\n".join([r.payload['text'] for r in best_results])
        risposta = ask_gemma(context, query)
        print(f"\n🤖 AI:\n{risposta}")

if __name__ == "__main__":
    start_simple_rag()