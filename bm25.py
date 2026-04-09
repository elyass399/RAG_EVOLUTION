import os, subprocess, json, time, re
from pathlib import Path
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi

# --- ENV & PROXY ---
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

# --- CLIENT & MODELLI ---
client = QdrantClient(url="http://localhost:6333")
COLLECTION = "zucchetti_knowledge_base" 
LLM_MODEL = "gemma3:27b-it-qat"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")

SYSTEM_PROMPT = """Sei l'Assistente RAG Ultra-Preciso. 
Usa i documenti forniti per rispondere. Se trovi dettagli su battaglie, persone o regole, riportali.
Se l'informazione non è presente, rispondi: "info non esiste"."""

# --- POWERSHELL BRIDGE ---
def _ps_invoke(url_suffix, payload):
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL").rstrip('/')
    url = f"{base_url}/{url_suffix}"
    stamp = str(time.time()).replace('.', '')
    req_file = Path(f"req_{stamp}.json").absolute().as_posix()
    res_file = Path(f"res_{stamp}.bin").absolute().as_posix()
    with open(req_file, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False)
    ps_cmd = f'$url="{url}"; $headers=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json;charset=utf-8"}}; try{{$body=[System.IO.File]::ReadAllBytes("{req_file}");$res=Invoke-WebRequest -Uri $url -Method Post -Headers $headers -Body $body -Proxy $null -UseBasicParsing -TimeoutSec 60;[System.IO.File]::WriteAllBytes("{res_file}", $res.RawContentStream.ToArray())}} catch{{$errMsg="ERROR: "+$_.Exception.Message;[System.IO.File]::WriteAllBytes("{res_file}", [System.Text.Encoding]::UTF8.GetBytes($errMsg))}}'
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

# --- EMBEDDING ---
#Richiama il modello embedding e restituisce un vettore numerico.
def get_embedding(text):
    data = _ps_invoke("embeddings", {"model": EMBED_MODEL, "input": text})
    return np.array(data['data'][0]['embedding']).flatten().tolist() if data else None

# --- LLM CALL ---
#Richiama il modello LLM con un prompt di sistema e una domanda utente, restituendo la risposta testuale.
def call_llm(sys, usr):
    payload = {"model": LLM_MODEL, "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}], "temperature": 0}
    data = _ps_invoke("chat/completions", payload)
    if data and 'choices' in data:
        return re.sub(r'</?end_of_turn>', '', data['choices'][0]['message']['content']).strip()
    return None

# --- ESPANSIONE QUERY ---
#Se la query è una sola parola, la arricchisce per aiutare BM25 e LLM.
def expand_query(text):
    if len(text.split()) == 1:
        return f"Informazioni e dettagli storici o tecnici su {text}"
    return text

# --- ROUTING MANUALE ---
#Usa LLM per capire a quale manuale appartiene la domanda, migliorando la ricerca.
def get_target_manual(query):
    prompt = f"Data la domanda: '{query}', scegli il manuale: 'calcio', 'guerre', 'prompt'. Rispondi solo con la parola."
    res = call_llm("Sei un classificatore.", prompt)
    if res:
        res = res.lower()
        for m in ['calcio', 'guerre', 'prompt']:
            if m in res: return m
    return None

# --- RICERCA BM25 + SEMANTIC RERANK ---
def hybrid_bm25_rerank(query_text, query_vec, manual):
    # 1. Filtro manuale
    q_filter = Filter(must=[FieldCondition(key="manual", match=MatchValue(value=manual))]) if manual else None
    
    # 2. Recupera documenti filtrati
    all_docs = client.scroll(COLLECTION, scroll_filter=q_filter, limit=500, with_payload=True)[0]
    if not all_docs: return []

    # 3. BM25
    # Creiamo un corpus testuale per BM25 e calcoliamo i punteggi
    corpus = [d.payload['text'] for d in all_docs]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query_text.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # 4. Semantic search
    #Ricerca semantica con embedding e salva score.
    semantic_results = client.query_points(COLLECTION, query=query_vec, query_filter=q_filter, limit=20).points
    semantic_scores = {r.id: r.score for r in semantic_results}

    # 5. Rerank combinato
    #Combina BM25 + semantico con pesi e crea lista di tuple (score, doc), poi ordina e prendi top 15.
    combined = []
    for idx, doc in enumerate(all_docs):
        doc_id = doc.id
        bm25_score = bm25_scores[idx]
        semantic_score = semantic_scores.get(doc_id, 0)
        final_score = 0.4 * bm25_score + 0.6 * semantic_score  # pesi: semantico più importante
        combined.append((final_score, doc))

    combined.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in combined[:15]]

    return top_docs

# --- CHAT LOOP ---
#Loop principale: prende input utente → espande query → identifica manuale → embedding → ricerca + rerank → genera risposta con LLM.
def start_chat():
    print(f"\n🚀 ZUCCHETTI RAG - BM25 + Semantic Rerank")
    while True:
        user_input = input("\n👤 Tu: ").strip()
        if user_input.lower() in ["esci", "quit"]: break
        if not user_input: continue

        rich_query = expand_query(user_input)
        manual_filter = get_target_manual(rich_query)
        print(f"   🎯 Manuale rilevato: {manual_filter}")

        query_vec = get_embedding(rich_query)
        results = hybrid_bm25_rerank(user_input, query_vec, manual_filter)

        if results:
            context_txt = "\n\n".join([f"--- DOC [ID {i}] ---\n{r.payload['text']}" for i, r in enumerate(results)])
            risposta = call_llm(SYSTEM_PROMPT, f"DOMANDA: {user_input}\n\nCONTESTO:\n{context_txt}")
            print(f"\n🤖 Gemma3:\n{risposta}")
        else:
            print("🤖 info non esiste")

if __name__ == "__main__":
    start_chat()