import os, json, asyncio, time, subprocess, re
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"
client = QdrantClient(url="http://localhost:6333")
COLLECTION = "parentchild"

MODEL_LLM = os.getenv("MODEL_SUMMARY", "gemma4:e4b")
MODEL_SLM = os.getenv("MODEL_SLM", "gemma4:e4b")
MODEL_EMB = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

def _ps_invoke_sync(url_suffix, payload):
    url = f"{os.getenv('LITELLM_BASE_URL').rstrip('/')}/{url_suffix}"
    api_key = os.getenv("LITELLM_API_KEY")
    stamp = f"{int(time.time() * 1000)}"
    req_f, res_f = Path(f"req_p_{stamp}.json").absolute(), Path(f"res_p_{stamp}.bin").absolute()
    try:
        with open(req_f, "w", encoding="utf-8") as f: json.dump(payload, f)
        ps_cmd = f'$u="{url}"; $h=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json"}}; $b=[System.IO.File]::ReadAllBytes("{req_f}"); try{{$r=Invoke-WebRequest -Uri $u -Method Post -Headers $h -Body $b -Proxy $null -UseBasicParsing -TimeoutSec 120; [System.IO.File]::WriteAllBytes("{res_f}", $r.RawContentStream.ToArray())}} catch{{}}'
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
        if os.path.exists(res_f):
            with open(res_f, "rb") as f: return json.loads(f.read().decode("utf-8-sig"))
    finally:
        if os.path.exists(req_f): os.remove(req_f)
        if os.path.exists(res_f): os.remove(res_f)
    return None

# --- 1. ROUTER: IDENTIFICA IL MANUALE ---
async def get_target_manual(query):
    """Chiede all'SLM di classificare la domanda per filtrare il database."""
    prompt = f"""Analizza la domanda dell'utente e decidi quale manuale consultare.
    Domanda: '{query}'
    
    Categorie disponibili:
    - 'calcio' (regole, arbitro, pallone, campo)
    - 'guerre' (storia, Roma, Cartagine, Annibale, Scipione)
    - 'prompt' (IA, prompt engineering, LLM)
    
    Rispondi SOLO con la parola chiave della categoria. Se incerto, rispondi 'tutti'."""
    
    res = await asyncio.to_thread(_ps_invoke_sync, "chat/completions", {
        "model": MODEL_SLM, "messages": [{"role": "user", "content": prompt}], "temperature": 0
    })
    if res:
        choice = res['choices'][0]['message']['content'].lower()
        if 'calcio' in choice: return "le17regoledelcalcio.md"
        if 'guerre' in choice: return "Storia-Le-guerre-puniche.md"
        if 'prompt' in choice: return "prompt_engineering.md"
    return None

# --- 2. CHAT ENGINE ---
async def start_chat():
    print(f"\n" + "="*60)
    print("🚀 ZUCCHETTI RAG: PRECISION PARENT-CHILD (Small-to-Big)")
    print("Logica: Routing Semantico + Recupero Gerarchico")
    print("="*60 + "\n")
    
    while True:
        user_input = input("👤 Tu: ").strip()
        if user_input.lower() in ["esci", "quit"]: break
        if not user_input: continue

        start_time = time.time()

        # A. ROUTING: Capiamo in quale manuale cercare
        target_file = await get_target_manual(user_input)
        print(f"🎯 Router: Filtro attivato su -> {target_file if target_file else 'Ricerca Globale'}")

        # B. EMBEDDING
        res_emb = await asyncio.to_thread(_ps_invoke_sync, "embeddings", {"model": MODEL_EMB, "input": user_input})
        q_vec = res_emb['data'][0]['embedding']

        # C. RETRIEVAL FILTRATO (Solo i figli del manuale giusto)
        q_filter = None
        if target_file:
            q_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=target_file))])

        hits = client.query_points(COLLECTION, query=q_vec, query_filter=q_filter, limit=5).points

        if not hits:
            print("🤖 AI: Nessuna informazione trovata nel manuale selezionato."); continue

        # D. RECUPERO DEI PADRI (Deduplicati)
        parents_to_read = {}
        for h in hits:
            p_text = h.payload.get('parent_text', '')
            source = h.payload.get('source', 'Sconosciuto')
            if p_text not in parents_to_read:
                parents_to_read[p_text] = source
        
        context = "\n\n".join([f"--- DOCUMENTO: {src} ---\n{txt}" for txt, src in parents_to_read.items()])

        # E. GENERAZIONE FINALE
        print("✍️  Generazione risposta di alta precisione...")
        sys_prompt = "Sei un analista tecnico. Rispondi alla domanda usando SOLO il contesto dei Padri fornito. Sii molto dettagliato."
        
        res_llm = await asyncio.to_thread(_ps_invoke_sync, "chat/completions", {
            "model": MODEL_LLM,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"CONTESTO:\n{context}\n\nDOMANDA: {user_input}"}],
            "temperature": 0.1
        })

        if res_llm:
            print(f"\n🤖 AI:\n{res_llm['choices'][0]['message']['content']}")
            print(f"\n⏱️ Tempo totale: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(start_chat())