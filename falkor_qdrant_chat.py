import os, subprocess, json, time, re
from pathlib import Path
from qdrant_client import QdrantClient
from falkordb import FalkorDB
from dotenv import load_dotenv

load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

# --- CONFIGURAZIONE ---
qdrant = QdrantClient(url="http://localhost:6333")
falkor = FalkorDB(host='localhost', port=6379)
graph = falkor.select_graph('ZucchettiKnowledgeGraph')
COLLECTION = "zucchetti_knowledge_base"
LLM_MODEL = "gemma3:27b-it-qat"

def _ps_invoke(url_suffix, payload):
    api_key = os.getenv("LITELLM_API_KEY")
    url = f"{os.getenv('LITELLM_BASE_URL').rstrip('/')}/{url_suffix}"
    stamp = str(time.time()).replace('.', '')
    req_file = Path(f"req_{stamp}.json").absolute().as_posix()
    res_file = Path(f"res_{stamp}.bin").absolute().as_posix()
    
    with open(req_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    
    # TIMEOUT PORTATO A 240 SECONDI
    ps_cmd = f'$u="{url}"; $h=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json"}}; try{{$b=[System.IO.File]::ReadAllBytes("{req_file}");$r=Invoke-WebRequest -Uri $u -Method Post -Headers $h -Body $b -Proxy $null -UseBasicParsing -TimeoutSec 240;[System.IO.File]::WriteAllBytes("{res_file}", $r.RawContentStream.ToArray())}} catch{{}}'
    
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    
    data = None
    if os.path.exists(res_file):
        with open(res_file, "rb") as f:
            try: data = json.loads(f.read().decode('utf-8-sig'))
            except: data = None
        os.remove(req_file); os.remove(res_file)
    return data

def get_local_entities(text):
    """Estrazione locale immediata"""
    words = re.findall(r'\b[A-Z][a-z]{2,}\b|\b[A-Z]{2,}\b|\b\w{7,}\b', text)
    ignore = ["DOMANDA", "INFORMAZIONI", "SPIEGAZIONE", "ZUCCHETTI", "MANUALE", "ANALOGIE"]
    return list(set([w.upper() for w in words if w.upper() not in ignore]))

def get_graph_facts(query):
    entities = get_local_entities(query)
    facts = []
    for ent in entities:
        if len(ent) < 3: continue
        q = f"MATCH (n:Concept) WHERE n.name CONTAINS '{ent}' MATCH (n)-[r1]->(m) OPTIONAL MATCH (m)-[r2]->(k) RETURN n.name, type(r1), m.name, type(r2), k.name LIMIT 8"
        try:
            res_graph = graph.query(q)
            for row in res_graph.result_set:
                facts.append(f"({row[0]}) {row[1]} ({row[2]})")
                if row[3]: facts.append(f"({row[2]}) {row[3]} ({row[4]})")
        except: continue
    return list(set(facts))[:15]

def start_chat():
    print(f"\n🚀 ZUCCHETTI GraphRAG v4.3 - MODALITÀ RESILIENZA")
    while True:
        user_input = input("👤 Tu: ").strip()
        if user_input.lower() in ["esci", "quit"]: break

        # 1. CHIAMATA: Embedding
        emb_res = _ps_invoke("embeddings", {"model": os.getenv("EMBEDDING_MODEL"), "input": user_input})
        if not emb_res:
            print("❌ Errore di rete (Embedding)."); continue
        vec = emb_res['data'][0]['embedding']
        
        # 2. RICERCA LOCALE
        search_res = qdrant.query_points(COLLECTION, query=vec, limit=3).points
        facts = get_graph_facts(user_input)

        # 3. CHIAMATA: Generazione con pacchetto ridotto (Max 4000 char)
        text_context = "\n".join([f"- {r.payload.get('text', '')[:800]}" for r in search_res])
        graph_context = "\n".join(facts) if facts else "Nessun legame certo."

        sys_prompt = "Sei l'Assistente GraphRAG. Usa il GRAFO per i nomi e i DOCUMENTI per i dettagli. Sii breve e preciso."
        user_prompt = f"GRAFO:\n{graph_context}\n\nTESTO:\n{text_context}\n\nDOMANDA: {user_input}"
        
        print(" Analisi in corso (Attesa stimata: 30-60s)...")
        
        risposta = _ps_invoke("chat/completions", {
            "model": LLM_MODEL, 
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt[:4000]}],
            "temperature": 0
        })
        
        if risposta and 'choices' in risposta:
            print(f"\n Gemma3:\n{risposta['choices'][0]['message']['content']}\n")
        else:
            print("\n Errore: Il server è troppo carico o la risposta è troppo complessa. Riprova con una domanda più specifica.")

if __name__ == "__main__":
    start_chat()