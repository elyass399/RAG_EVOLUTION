import os, subprocess, json, time, re
from pathlib import Path
from falkordb import FalkorDB
from dotenv import load_dotenv

# --- 1. SETUP AMBIENTE ---
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

print("\n" + "="*60)
print("🚀 AVVIO INGESTIONE: RESET & RICOSTRUZIONE TOTALE (6380)")
print("="*60)

# Connessione al container FalkorDB Pure (Porta 6380)
try:
    db = FalkorDB(host='localhost', port=6380)
    graph = db.select_graph('ZucchettiPureGraph')
    print("✅ Connesso a FalkorDB sulla porta 6380.")
except Exception as e:
    print(f"❌ Errore connessione: {e}")
    exit()

CHUNKS_DIR = Path("./output_late_chunking")
LLM_MODEL = "gemma3:27b-it-qat"

# --- HELPERS DI PULIZIA ---
def clean_node_name(text):
    """Pulisce Soggetto e Oggetto"""
    if not text: return "IGNOTO"
    return str(text).replace("'", "").replace('"', '').replace("\\", "").strip().upper()

def clean_label(text):
    """Pulisce la Relazione (evita il crash -[:]->)"""
    if not text: return "CORRELATO_A"
    text = text.upper().replace(" ", "_")
    clean = re.sub(r'[^A-Z0-9_]', '', text)
    return clean if clean else "CORRELATO_A"

# --- BRIDGE POWERSHELL ---
def _ps_invoke_chat(prompt):
    api_key = os.getenv("LITELLM_API_KEY")
    url = f"{os.getenv('LITELLM_BASE_URL').rstrip('/')}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Sei un esperto di grafi. Estrai le relazioni. Rispondi SOLO JSON: [[S, R, O], ...]"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    stamp = str(time.time()).replace('.', '')
    req_file = f"req_pure_{stamp}.json"
    res_file = f"res_pure_{stamp}.bin"
    with open(req_file, "w", encoding="utf-8") as f: json.dump(payload, f)
    
    ps_cmd = f'$u="{url}"; $h=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json"}}; $b=[System.IO.File]::ReadAllBytes("{req_file}"); try{{$r=Invoke-WebRequest -Uri $u -Method Post -Headers $h -Body $b -Proxy $null -UseBasicParsing -TimeoutSec 120; [System.IO.File]::WriteAllBytes("{res_file}", $r.RawContentStream.ToArray())}} catch{{}}'
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    
    data = None
    if os.path.exists(res_file):
        with open(res_file, "rb") as f: 
            try: data = json.loads(f.read().decode('utf-8-sig'))
            except: data = None
        os.remove(req_file); os.remove(res_file)
    return data

# --- LOGICA DI RICOSTRUZIONE ---
def run_pure_ingestion():
    # --- RESET DEL DATABASE ---
    print("🧨 [RESET] Eliminazione del grafo esistente...")
    try:
        graph.delete()
        print("✅ Database pulito.")
    except:
        print("ℹ️ Il database era già vuoto.")

    files = sorted(list(CHUNKS_DIR.glob("*.json")))
    print(f"📂 [START] Inizio elaborazione completa di {len(files)} file...")

    for f_path in files:
        print(f"\n📄 Analisi file: {f_path.name}")
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data.get('text', '')
            source = data.get('source', 'Unknown')
            chunk_id = f_path.stem

            # 1. Crea il nodo CHUNK (Contenitore del testo)
            text_safe = clean_node_name(text)
            q_chunk = f"CREATE (:Chunk {{id: '{chunk_id}', content: '{text_safe}', source: '{source}'}})"
            graph.query(q_chunk)

            # 2. Estrazione Triplette via Gemma3
            prompt = f"Estrai triplette JSON [[S, R, O]] dal testo tecnico Zucchetti:\n{text[:3000]}"
            res = _ps_invoke_chat(prompt)
            
            if res and 'choices' in res:
                content = res['choices'][0]['message']['content']
                match = re.search(r'\[\s*\[.*\]\s*\]', content, re.DOTALL)
                if match:
                    # Riparazione manuale del JSON per errori comuni degli LLM
                    json_str = match.group().replace('] [', '], [').replace('][', '],[')
                    try:
                        triplets = json.loads(json_str)
                        print(f"   📊 Trovate {len(triplets)} relazioni.")
                        
                        for s, r, o in triplets:
                            s_name = clean_node_name(s)
                            r_type = clean_label(r)
                            o_name = clean_node_name(o)

                            if not s_name or not o_name: continue

                            # 3. Disegna nodi e archi relazionali
                            query_rel = f"""
                            MERGE (c1:Concept {{name: '{s_name}'}})
                            MERGE (c2:Concept {{name: '{o_name}'}})
                            MERGE (c1)-[:{r_type}]->(c2)
                            WITH c1, c2
                            MATCH (ch:Chunk {{id: '{chunk_id}'}})
                            MERGE (ch)-[:CONTIENE]->(c1)
                            MERGE (ch)-[:CONTIENE]->(c2)
                            """
                            try:
                                graph.query(query_rel)
                            except Exception as e:
                                print(f"      ❌ Errore Cypher (saltata): {e}")
                                
                    except Exception as e:
                        print(f"   ⚠️ Errore nel parse JSON delle triplette: {e}")
            else:
                print(f"   ❌ Il server di Padova non ha risposto per {f_path.name}")

    print("\n🏆 RICOSTRUZIONE COMPLETATA!")

if __name__ == "__main__":
    run_pure_ingestion()