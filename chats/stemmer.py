import os, subprocess, json, time, re, string
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem import SnowballStemmer

# --- SETUP ---
try:
    stemmer = SnowballStemmer("italian")
except:
    nltk.download('punkt')
    stemmer = SnowballStemmer("italian")

load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "zucchetti_knowledge_base" 
LLM_MODEL = "gemma3:27b-it-qat"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

SYSTEM_PROMPT = """Sei l'Assistente RAG Ultra-Preciso.
Rispondi alla domanda usando SOLO il contesto fornito. 
Se trovi tabelle o elenchi, formattali bene. 
Se l'informazione non c'è, dì: "info non esiste"."""

def _ps_invoke(url_suffix, payload):
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL").rstrip('/')
    url = f"{base_url}/{url_suffix}"
    stamp = str(time.time()).replace('.', '')
    req_file = Path(f"req_{stamp}.json").absolute().as_posix()
    res_file = Path(f"res_{stamp}.bin").absolute().as_posix()
    with open(req_file, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False)
    ps_cmd = f'$url="{url}"; $headers=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json;charset=utf-8"}}; try{{$body=[System.IO.File]::ReadAllBytes("{req_file}");$res=Invoke-WebRequest -Uri $url -Method Post -Headers $headers -Body $body -Proxy $null -UseBasicParsing -TimeoutSec 120;[System.IO.File]::WriteAllBytes("{res_file}", $res.RawContentStream.ToArray())}} catch{{$errMsg="ERROR: "+$_.Exception.Message;[System.IO.File]::WriteAllBytes("{res_file}", [System.Text.Encoding]::UTF8.GetBytes($errMsg))}}'
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    data = None
    if os.path.exists(res_file):
        with open(res_file, "rb") as f: raw = f.read()
        if not raw.startswith(b"ERROR"):
            try: data = json.loads(raw.decode('utf-8-sig'))
            except: data = None
    if os.path.exists(req_file): os.remove(req_file)
    if os.path.exists(res_file): os.remove(res_file)
    return data

def preprocess_italian(text):
    if not text: return []
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return [stemmer.stem(word) for word in text.split()]

def get_embedding(text):
    data = _ps_invoke("embeddings", {"model": EMBED_MODEL, "input": text})
    return data['data'][0]['embedding'] if data else None

def call_llm(sys, usr):
    # Tronchiamo l'input utente (contesto + domanda) a circa 12.000 caratteri per sicurezza server
    payload = {"model": LLM_MODEL, "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr[:12000]}], "temperature": 0}
    data = _ps_invoke("chat/completions", payload)
    if data and 'choices' in data:
        return data['choices'][0]['message']['content'].strip()
    return "⚠️ Errore di timeout o risposta vuota dal server."

def get_target_manual(query):
    prompt = f"Analizza la domanda: '{query}'. Rispondi solo con una parola: 'calcio', 'guerre', 'prompt' o 'nessuno'."
    res = call_llm("Sei un classificatore.", prompt)
    if res:
        res = res.lower()
        for m in ['calcio', 'guerre', 'prompt']:
            if m in res: return m
    return None

def hybrid_search_final(query_text, query_vec, manual_tag):
    mapping = {
        "calcio": "le17regoledelcalcio.md",
        "guerre": "Storia-Le-guerre-puniche.md",
        "prompt": "prompt_engineering.md"
    }
    target_file = mapping.get(manual_tag)
    
    # Filtro flessibile
    q_filter = None
    if target_file:
        q_filter = Filter(should=[
            FieldCondition(key="source", match=MatchValue(value=target_file)),
            FieldCondition(key="manual", match=MatchValue(value=manual_tag))
        ])

    # 1. Recupero documenti
    all_docs = client.scroll(COLLECTION, scroll_filter=q_filter, limit=100, with_payload=True)[0]
    
    # Fallback globale se il filtro manuale fallisce
    if not all_docs:
        all_docs = client.scroll(COLLECTION, limit=100, with_payload=True)[0]

    if not all_docs: return []

    # 2. BM25 con Stemming
    corpus = [d.payload.get('text', d.payload.get('child_text', '')) for d in all_docs]
    tokenized_corpus = [preprocess_italian(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(preprocess_italian(query_text))

    # 3. Combinazione risultati
    combined = []
    for idx, doc in enumerate(all_docs):
        # Se il testo è troppo breve (<10 char), saltiamolo
        text = doc.payload.get('text', doc.payload.get('child_text', ''))
        if len(text) < 10: continue

        b_score = bm25_scores[idx]
        boost = 10.0 if manual_tag and manual_tag in doc.payload.get("source", "").lower() else 0.0
        
        combined.append((b_score + boost, doc))

    combined.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in combined[:6]] # Prendiamo solo i top 6 per non appesantire l'LLM

def start_chat():
    print(f"\n🚀 ZUCCHETTI RAG - SISTEMA OTTIMIZZATO")
    while True:
        user_input = input("\n👤 Tu: ").strip()
        if user_input.lower() in ["esci", "quit"]: break
        if not user_input: continue

        manual_tag = get_target_manual(user_input)
        print(f"   🎯 Manuale rilevato: {manual_tag}")

        query_vec = get_embedding(user_input)
        results = hybrid_search_final(user_input, query_vec, manual_tag)

        if results:
            print(f"   📂 Trovati {len(results)} documenti rilevanti.")
            context_blocks = []
            for i, r in enumerate(results):
                # Usiamo sia il contesto globale (Gemma) che il testo del chunk
                g_ctx = r.payload.get('context', 'N/A')
                txt = r.payload.get('text', r.payload.get('child_text', ''))
                context_blocks.append(f"--- DOC {i} [Ambito: {g_ctx}] ---\n{txt}")
            
            context_txt = "\n\n".join(context_blocks)
            risposta = call_llm(SYSTEM_PROMPT, f"CONTESTO:\n{context_txt}\n\nDOMANDA: {user_input}")
            print(f"\n🤖 Gemma3:\n{risposta}")
        else:
            print("🤖 info non esiste")

if __name__ == "__main__":
    start_chat()