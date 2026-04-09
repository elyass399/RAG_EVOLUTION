import os, subprocess, json, time, re
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi

# --- ENV ---
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "manuali_late_chunking"
LLM_MODEL = "gemma3:27b-it-qat"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")

SYSTEM_PROMPT = """Sei un assistente RAG ultra-preciso.
Rispondi SOLO usando il contesto fornito.
Se l'informazione non è presente: "info non esiste".
Cita il numero del documento quando possibile."""

# --- POWERSHELL ---
def _ps_invoke(url_suffix, payload):
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL").rstrip('/')
    url = f"{base_url}/{url_suffix}"
    stamp = str(time.time()).replace('.', '')
    req_file = Path(f"req_{stamp}.json").absolute().as_posix()
    res_file = Path(f"res_{stamp}.bin").absolute().as_posix()

    with open(req_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    ps_cmd = f'$url="{url}"; $headers=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json;charset=utf-8"}}; try{{$body=[System.IO.File]::ReadAllBytes("{req_file}");$res=Invoke-WebRequest -Uri $url -Method Post -Headers $headers -Body $body -Proxy $null -UseBasicParsing -TimeoutSec 60;[System.IO.File]::WriteAllBytes("{res_file}", $res.RawContentStream.ToArray())}} catch{{$errMsg="ERROR: "+$_.Exception.Message;[System.IO.File]::WriteAllBytes("{res_file}", [System.Text.Encoding]::UTF8.GetBytes($errMsg))}}'

    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)

    if not os.path.exists(res_file):
        return None

    with open(res_file, "rb") as f:
        raw = f.read()

    os.remove(req_file)
    os.remove(res_file)

    if raw.startswith(b"ERROR"):
        return None

    try:
        return json.loads(raw.decode('utf-8-sig'))
    except:
        return None


# --- EMBEDDING ---
def get_embedding(text):
    data = _ps_invoke("embeddings", {"model": EMBED_MODEL, "input": text})
    if not data:
        return None
    return np.array(data['data'][0]['embedding']).flatten().tolist()


# --- LLM ---
def call_llm(sys, usr):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr}
        ],
        "temperature": 0
    }
    data = _ps_invoke("chat/completions", payload)
    if data and 'choices' in data:
        return re.sub(r'</?end_of_turn>', '', data['choices'][0]['message']['content']).strip()
    return "errore LLM"


# --- QUERY EXPANSION ---
def expand_query(text):
    if len(text.split()) <= 2:
        return f"Spiegazione dettagliata e informazioni su {text}"
    return text


# --- CLASSIFICATION (SOFT USE) ---
def get_target_manual(query):
    prompt = f"Domanda: '{query}'. Categoria: calcio, guerre, prompt. Rispondi solo con una parola."
    res = call_llm("Sei un classificatore.", prompt)
    if res:
        res = res.lower()
        for m in ['calcio', 'guerre', 'prompt']:
            if m in res:
                return m
    return None


# --- NORMALIZATION (FIXED BUG) ---
def normalize(scores):
    if scores is None or len(scores) == 0:
        return scores

    scores = np.array(scores)

    min_s = np.min(scores)
    max_s = np.max(scores)

    if max_s - min_s == 0:
        return [0.5] * len(scores)

    return ((scores - min_s) / (max_s - min_s)).tolist()


# --- HYBRID SEARCH ---
def hybrid_search(query_text, query_vec, manual):

    # 1. Vector search first (reduce scope)
    semantic_results = client.query_points(
        COLLECTION,
        query=query_vec,
        limit=100
    ).points

    if not semantic_results:
        return []

    docs = []
    texts = []
    semantic_scores = []

    for r in semantic_results:
        docs.append(r)
        texts.append(r.payload.get("text", ""))
        semantic_scores.append(r.score)

    # 2. BM25 on reduced set
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query_text.lower().split())

    # 3. Normalize
    bm25_norm = normalize(bm25_scores)
    semantic_norm = normalize(semantic_scores)

    # 4. Combine scores + soft boost
    combined = []

    for i, doc in enumerate(docs):
        score = 0.6 * semantic_norm[i] + 0.4 * bm25_norm[i]

        # Soft boost if manual matches
        if manual and doc.payload.get("manual") == manual:
            score += 0.15

        combined.append((score, doc))

    combined.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in combined[:8]]  # limit context


# --- CHAT LOOP ---
def start_chat():
    print("\n🚀 RAG OPTIMIZED VERSION")

    while True:
        user_input = input("\n👤 Tu: ").strip()

        if user_input.lower() in ["esci", "quit"]:
            break

        if not user_input:
            continue

        query = expand_query(user_input)
        manual = get_target_manual(query)

        print(f"🎯 Categoria stimata: {manual}")

        query_vec = get_embedding(query)

        if not query_vec:
            print("Errore embedding")
            continue

        # ⚠️ IMPORTANT FIX: use expanded query
        results = hybrid_search(query, query_vec, manual)

        if not results:
            print("🤖 info non esiste")
            continue

        # Build clean context
        context_blocks = []
        for i, r in enumerate(results):
            text = r.payload.get("text", "")
            context_blocks.append(f"Documento {i+1}:\n{text}")

        context = "\n\n".join(context_blocks)

        prompt = f"""
DOMANDA:
{user_input}

CONTESTO:
{context}
"""

        answer = call_llm(SYSTEM_PROMPT, prompt)
        print(f"\n🤖 Risposta:\n{answer}")


if __name__ == "__main__":
    start_chat()