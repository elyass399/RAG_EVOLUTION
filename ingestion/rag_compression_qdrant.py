import os, json, time, re, subprocess, asyncio
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "manuali_late_chunking" 

MODEL_LLM = os.getenv("MODEL_SUMMARY", "gemma4:26b")
MODEL_SLM = os.getenv("MODEL_SLM", "gemma4:e4b") # Usiamo il piccolo per il routing veloce
MODEL_EMB = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

def _ps_invoke_sync(url_suffix, payload):
    url = f"{os.getenv('LITELLM_BASE_URL').rstrip('/')}/{url_suffix}"
    api_key = os.getenv("LITELLM_API_KEY")
    stamp = f"{int(time.time() * 1000)}"
    req_f, res_f = Path(f"req_chat_{stamp}.json").absolute(), Path(f"res_chat_{stamp}.bin").absolute()
    with open(req_f, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False)
    ps_cmd = (f'$u="{url}"; $h=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json"}}; '
              f'$b=[System.IO.File]::ReadAllBytes("{req_f}"); try{{$r=Invoke-WebRequest -Uri $u -Method Post '
              f'-Headers $h -Body $b -Proxy $null -UseBasicParsing -TimeoutSec 120; '
              f'[System.IO.File]::WriteAllBytes("{res_f}", $r.RawContentStream.ToArray())}} catch{{}}')
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    data = None
    if os.path.exists(res_f):
        with open(res_f, "rb") as f: 
            try: data = json.loads(f.read().decode("utf-8-sig"))
            except: data = None
        os.remove(req_f); os.remove(res_f)
    return data

async def ai_call(url_suffix, payload):
    return await asyncio.to_thread(_ps_invoke_sync, url_suffix, payload)

# --- 1. ROUTER: CAPISCE IL MANUALE TARGET ---
async def get_target_manual(query):
    """Analizza la domanda e decide quale manuale filtrare."""
    prompt = f"""Analizza la domanda dell'utente e decidi quale manuale consultare.
    Domanda: '{query}'
    
    Opzioni:
    - 'calcio' (se parla di regole di gioco, arbitri, pallone)
    - 'guerre' (se parla di storia, Roma, Cartagine, Annibale)
    - 'prompt' (se parla di intelligenza artificiale o prompt engineering)

    
    Rispondi con una sola parola (l'opzione scelta)."""
    
    res = await ai_call("chat/completions", {
        "model": MODEL_SLM, 
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    })
    if res:
        choice = res['choices'][0]['message']['content'].lower()
        if 'calcio' in choice: return "le17regoledelcalcio.md"
        if 'guerre' in choice: return "Storia-Le-guerre-puniche.md"
        if 'prompt' in choice: return "prompt_engineering.md"
    return None

# --- 2. CHAT ENGINE CON FILTRAGGIO METADATI ---
async def start_chat():
    print(f"\n" + "="*60)
    print("🚀 ZUCCHETTI RAG MASTER: SEMANTIC ROUTING ATTIVO")
    print("="*60 + "\n")
    
    while True:
        user_input = input("👤 Tu: ").strip()
        if user_input.lower() in ["esci", "quit"]: break
        if not user_input: continue

        start_time = time.time()

        # STEP A: ROUTING (Capiamo dove cercare)
        target_file = await get_target_manual(user_input)
        print(f"🎯 Router: Direzione -> {target_file if target_file else 'Tutta la Knowledge Base'}")

        # STEP B: EMBEDDING
        res_emb = await ai_call("embeddings", {"model": MODEL_EMB, "input": user_input})
        q_vec = res_emb['data'][0]['embedding']

        # STEP C: RETRIEVAL FILTRATO (Rimuove il rumore del calcio dalla storia)
        q_filter = None
        if target_file:
            q_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=target_file))])

        hits = client.query_points(COLLECTION, query=q_vec, query_filter=q_filter, limit=5).points

        # STEP D: CONTEXT BUILDING
        context_list = []
        for i, h in enumerate(hits):
            g_ctx = h.payload.get('context', '')
            l_txt = h.payload.get('text', '')
            context_list.append(f"--- DOCUMENTO {i} ---\nAMBITO: {g_ctx}\nCONTENUTO: {l_txt}")
        
        full_context = "\n\n".join(context_list)

        # STEP E: GENERAZIONE
        print("✍️  Sintesi in corso...")
        sys_prompt = "Sei un analista esperto Zucchetti. Rispondi usando il contesto fornito. Se il contesto non contiene la risposta, ammettilo."
        res_llm = await ai_call("chat/completions", {
            "model": MODEL_LLM, 
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"CONTESTO:\n{full_context}\n\nDOMANDA: {user_input}"}],
            "temperature": 0.1
        })

        if res_llm:
            print(f"\n🤖 AI:\n{res_llm['choices'][0]['message']['content']}")
            print(f"\n⏱️ Tempo risposta: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(start_chat())