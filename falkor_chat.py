import os, subprocess, json, time, re
from pathlib import Path
from falkordb import FalkorDB
from dotenv import load_dotenv

load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

db = FalkorDB(host='localhost', port=6380)
graph = db.select_graph('ZucchettiPureGraph')
LLM_MODEL = "gemma3:27b-it-qat"

def _ps_invoke(url_suffix, payload):
    api_key = os.getenv("LITELLM_API_KEY")
    url = f"{os.getenv('LITELLM_BASE_URL').rstrip('/')}/{url_suffix}"
    stamp = str(time.time()).replace('.', '')
    req_file = Path(f"req_{stamp}.json").absolute().as_posix()
    res_file = Path(f"res_{stamp}.bin").absolute().as_posix()
    with open(req_file, "w", encoding="utf-8") as f: json.dump(payload, f)
    ps_cmd = f'$u="{url}"; $h=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json"}}; try{{$b=[System.IO.File]::ReadAllBytes("{req_file}");$r=Invoke-WebRequest -Uri $u -Method Post -Headers $h -Body $b -Proxy $null -UseBasicParsing -TimeoutSec 120; [System.IO.File]::WriteAllBytes("{res_file}", $r.RawContentStream.ToArray())}} catch{{}}'
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    data = None
    if os.path.exists(res_file):
        with open(res_file, "rb") as f: 
            try: data = json.loads(f.read().decode('utf-8-sig'))
            except: data = None
        os.remove(req_file); os.remove(res_file)
    return data

# --- LOGICA UNIVERSALE DI RECUPERO ---

def get_graph_context(query):
    """
    Algoritmo di Scoring per Intersezione:
    Trova i chunk che coprono il maggior numero di concetti della domanda.
    """
    # 1. Estrazione Keywords (solo parole piene > 3 lettere)
    raw_words = re.findall(r'\w{4,}', query.upper())
    ignore = ["DOMANDA", "QUESTO", "QUALE", "QUANDO", "SISTEMA", "ZUCCHETTI", "INFORMAZIONI", "PERIODO", "DURATA"]
    keywords = [w for w in raw_words if w not in ignore]
    
    if not keywords: # Fallback se la domanda è cortissima
        keywords = re.findall(r'\w{3,}', query.upper())

    print(f"🔎 [DEBUG] Analisi multivariata per: {keywords}")

    # 2. Recupero Chunk e calcolo punteggio di rilevanza
    # Cerchiamo tutti i chunk che contengono ALMENO una keyword
    candidate_chunks = {} # Testo -> Punteggio
    
    for kw in keywords:
        q = f"MATCH (ch:Chunk) WHERE ch.content CONTAINS '{kw}' RETURN ch.content LIMIT 10"
        res = graph.query(q)
        for row in res.result_set:
            text = row[0]
            if text not in candidate_chunks:
                candidate_chunks[text] = 0
            # Aumentiamo il punteggio per ogni keyword diversa trovata nel testo
            candidate_chunks[text] += 1

    # Ordiniamo i chunk per punteggio (quelli che hanno più keyword in comune con la domanda)
    sorted_chunks = sorted(candidate_chunks.items(), key=lambda x: x[1], reverse=True)
    best_chunks = [c[0] for c in sorted_chunks[:5]] # Prendiamo i top 5

    # 3. Recupero Relazioni correlate ai Best Chunks
    facts = []
    seen_facts = set()
    for text in best_chunks:
        # Cerchiamo i concetti contenuti in questi specifici paragrafi
        q_rel = f"MATCH (ch:Chunk)-[:CONTIENE]->(c:Concept)-[r]->(m:Concept) WHERE ch.content = '{text[:100].replace("'", "")}...' RETURN c.name, type(r), m.name LIMIT 5"
        # Per semplicità cerchiamo relazioni basate sulle keywords nei nodi
        for kw in keywords:
            q_rel = f"MATCH (c:Concept) WHERE c.name CONTAINS '{kw}' MATCH (c)-[r]->(m:Concept) RETURN c.name, type(r), m.name LIMIT 5"
            res_rel = graph.query(q_rel)
            for row in res_rel.result_set:
                f = f"{row[0]} {row[1].replace('_', ' ')} {row[2]}"
                if f not in seen_facts:
                    facts.append(f)
                    seen_facts.add(f)

    # 4. Controllo universale per la presenza di DATI (Numeri/Anni)
    has_numbers = any(re.search(r'\d+', c) for c in best_chunks)
    print(f"📊 [DEBUG] Recuperati {len(best_chunks)} paragrafi e {len(facts)} fatti.")
    print(f"🛡️ [CHECK] Il contesto contiene dati numerici/date? {'SÌ' if has_numbers else 'NO'}")

    return facts[:20], best_chunks

def start_pure_chat():
    print("\n🚀 ZUCCHETTI PURE GRAPHRAG - UNIVERSAL LOGIC MODE\n")

    while True:
        user_input = input("👤 Tu: ").strip()
        if user_input.lower() in ["esci", "quit"]: break

        facts, chunks = get_graph_context(user_input)

        if not chunks:
            print("🤖 AI: INFO NON PRESENTE NEL GRAFO"); continue

        graph_txt = "\n".join(f"- {f}" for f in facts)
        docs_txt = "\n\n".join(chunks)

        # Prompt Universale: si adatta ad ogni manuale
        sys_prompt = """Sei l'assistente basato su GraphRAG.
        REGOLE:
        1. Rispondi usando il CONTESTO TESTUALE per i dettagli e le RELAZIONI per i fatti.
        2. Se la domanda richiede date, numeri o durate, estraili con cura dai paragrafi forniti.
        3. Non inventare nulla. Se i dati non ci sono, ammettilo.
        4. Sii professionale e sintetico."""

        user_prompt = f"CONTESTO TESTUALE:\n{docs_txt}\n\nRELAZIONI LOGICHE:\n{graph_txt}\n\nDOMANDA: {user_input}"

        print("🧐 Ragionamento multimodale...")
        res = _ps_invoke("chat/completions", {
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt[:15000]}],
            "temperature": 0
        })

        if res and 'choices' in res:
            print(f"\n🤖 Risposta:\n{res['choices'][0]['message']['content']}\n")

if __name__ == "__main__":
    start_pure_chat()