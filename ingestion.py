import os, subprocess, json, time, re
from pathlib import Path
from falkordb import FalkorDB
from dotenv import load_dotenv

# --- 1. SETUP AMBIENTE E LOGS ---
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

print("\n" + "="*60)
print("🚀 AVVIO MOTORE INGESTION GRAPHRAG (SLIDING WINDOW)")
print("="*60)

try:
    db = FalkorDB(host='localhost', port=6379)
    graph = db.select_graph('ZucchettiKnowledgeGraph')
    print("✅ [DB] Connesso a FalkorDB su localhost:6379")
except Exception as e:
    print(f"❌ [DB] Errore connessione FalkorDB: {e}")
    exit()

CHUNKS_DIR = Path("./output_late_chunking")
LLM_MODEL = "gemma3:27b-it-qat"

def clean_cypher_label(text):
    text = text.upper().replace(" ", "_").replace("'", "")
    clean = re.sub(r'[^A-Z0-9_]', '', text)
    return clean if clean else "CORRELATO_A"

def clean_node_content(text):
    return str(text).replace("'", "").replace('"', '').replace("\\", "").strip()

# --- 2. BRIDGE POWERSHELL ---
def _ps_invoke_chat(prompt):
    api_key = os.getenv("LITELLM_API_KEY")
    url = f"{os.getenv('LITELLM_BASE_URL').rstrip('/')}/chat/completions"
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Sei un estrattore di grafi. Rispondi SOLO in formato JSON: [[Soggetto, Relazione, Oggetto], ...]"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    
    stamp = str(time.time()).replace('.', '')
    req_file = f"greq_{stamp}.json"
    res_file = f"gres_{stamp}.bin"
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


# --- 3. LOGICA FINESTRA SCORREVOLE ---
def split_text_into_windows(text, window_size=3500, overlap=500):
    """Spezza testi giganti in finestre con sovrapposizione (overlap)"""
    windows = []
    start = 0
    while start < len(text):
        end = start + window_size
        windows.append(text[start:end])
        if end >= len(text):
            break
        start += (window_size - overlap) # Sposta la finestra togliendo l'overlap
    return windows


# --- 4. ESTRAZIONE TRIPLETTE ---
def extract_triplets_windowed(full_text, filename):
    # Se il testo è corto, facciamo una sola chiamata. Se è lungo lo spezziamo in finestre.
    windows = split_text_into_windows(full_text)
    all_triplets = []

    print(f"   🧠 [LLM] Analisi di {filename} (Spezzato in {len(windows)} finestre di lettura)...")

    for idx, window in enumerate(windows):
        if len(windows) > 1:
            print(f"      🔹 Analisi Finestra {idx+1}/{len(windows)} (lunghezza: {len(window)} caratteri)")

        prompt = f"""Estrai le relazioni tecniche principali dal testo. 
        Restituisci solo un array JSON di triplette.
        Esempio: [["PALLONE", "DEVE_ESSERE", "SFERICO"], ["SCIPIONE", "SCONFIGGE", "ANNIBALE"]]
        Testo: {window}"""
        
        data = _ps_invoke_chat(prompt)
        if data and 'choices' in data:
            content = data['choices'][0]['message']['content']
            try:
                match = re.search(r'\[\s*\[.*\]\s*\]', content, re.DOTALL)
                if match:
                    local_triplets = json.loads(match.group())
                    all_triplets.extend(local_triplets)
            except:
                pass
    
    print(f"      📊 Totale relazioni trovate per questo file: {len(all_triplets)}")
    return all_triplets

# --- 5. MOTORE DI INGESTION ---
def run_ingestion():
    try:
        graph.delete()
        print("\n🧨 [GRAFO] Svuotamento database per nuova ingestione.")
    except:
        pass

    files = sorted(list(CHUNKS_DIR.glob("*.json")))
    if not files:
        print(f"❌ [ERRORE] Nessun file JSON trovato in {CHUNKS_DIR}")
        return

    print(f"📂 [FILE] Inizio elaborazione di {len(files)} chunk...")

    for f_path in files:
        print(f"\n📄 Analisi file: {f_path.name}")
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data.get('text', '')
            source = data.get('payload', {}).get('source', 'Unknown')
            
            # Estrazione TRIPLETTE usando le finestre scorrevoli (Nessun limite di lunghezza!)
            triplets = extract_triplets_windowed(text, f_path.name)
            
            for item in triplets:
                if not isinstance(item, list) or len(item) < 3: continue
                
                s_name = clean_node_content(item[0]).upper()
                r_type = clean_cypher_label(item[1])
                o_name = clean_node_content(item[2]).upper()

                if not s_name or not o_name: continue

                query = f"""
                MERGE (c1:Concept {{name: '{s_name}'}})
                MERGE (c2:Concept {{name: '{o_name}'}})
                MERGE (c1)-[rel:{r_type}]->(c2)
                SET rel.source = '{source}'
                """
                try:
                    graph.query(query)
                except Exception as e:
                    pass # Evita di rompere il ciclo per un singolo nodo fallito

    print("\n" + "="*60)
    print("🏆 KNOWLEDGE GRAPH COMPLETATO SU FALKORDB!")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_ingestion()