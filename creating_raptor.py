import os, json, time, subprocess
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import warnings

# Disabilita i warning di scikit-learn se i cluster sono pochi
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.mixture import GaussianMixture

# ──────────────────────────────────────────────────────────────────────────
# SETUP E CONNESSIONE API
# ──────────────────────────────────────────────────────────────────────────
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

BASE_URL = os.getenv("LITELLM_BASE_URL", "").rstrip("/")
API_KEY  = os.getenv("LITELLM_API_KEY")
MODEL_LLM = os.getenv("MODEL_SUMMARY", "gemma4:26b")
MODEL_EMB = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

def _ps_invoke_sync(url_suffix, payload):
    """Tunnel PowerShell per i proxy aziendali."""
    url = f"{BASE_URL}/{url_suffix}"
    stamp = f"{int(time.time() * 1000)}_{os.getpid()}"
    req_f, res_f = Path(f"req_{stamp}.json").absolute(), Path(f"res_{stamp}.bin").absolute()
    try:
        with open(req_f, "w", encoding="utf-8") as f: 
            json.dump(payload, f, ensure_ascii=False)
        ps_cmd = (f'$u="{url}"; $h=@{{"Authorization"="Bearer {API_KEY}";"Content-Type"="application/json"}}; '
                  f'$b=[System.IO.File]::ReadAllBytes("{req_f}"); try{{$r=Invoke-WebRequest -Uri $u -Method Post '
                  f'-Headers $h -Body $b -Proxy $null -UseBasicParsing -TimeoutSec 240; '
                  f'[System.IO.File]::WriteAllBytes("{res_f}", $r.RawContentStream.ToArray())}} catch{{}}')
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
        if res_f.exists():
            with open(res_f, "rb") as f: 
                return json.loads(f.read().decode("utf-8-sig"))
    finally:
        for f in [req_f, res_f]:
            if f.exists(): f.unlink()
    return None

def get_embedding(text):
    """Restituisce l'embedding come lista nativa."""
    res = _ps_invoke_sync("embeddings", {"model": MODEL_EMB, "input": text[:8000]})
    return res['data'][0]['embedding'] if res else None

def get_summary(texts):
    """Genera il riassunto (LLM)."""
    prompt = "Sintetizza questi testi in un unico riassunto tecnico dettagliato. Mantieni date e numeri:\n\n---\n\n" + "\n\n---\n\n".join(texts)
    res = _ps_invoke_sync("chat/completions", {
        "model": MODEL_LLM, 
        "messages": [{"role": "user", "content": prompt[:15000]}], 
        "temperature": 0.1
    })
    return res['choices'][0]['message']['content'].strip() if res else "Riassunto non disponibile."

# ──────────────────────────────────────────────────────────────────────────
# MOTORE RAPTOR (STEP-BY-STEP FLAT DIRECTORY)
# ──────────────────────────────────────────────────────────────────────────
def run_raptor_step_by_step():
    print("\n" + "="*60)
    print("🚀 ZUCCHETTI RAPTOR: PROCEDURA STEP-BY-STEP (FLAT DIRECTORY)")
    print("="*60 + "\n")

    # 1. Selezionare cartelle (output_late_chunking -> output_raptor)
    INPUT_DIR = Path("./output_late_chunking").absolute()
    OUTPUT_DIR = Path("./output_raptor").absolute()
    
    if not INPUT_DIR.exists():
        print(f"❌ ERRORE: La cartella {INPUT_DIR} non esiste!")
        return
        
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Contatori globali per garantire ID unici su tutto il processo
    l1_counter = 1
    l2_counter = 1

    # 2. Lettura di tutti i file JSON e raggruppamento tramite attributo 'source'
    all_files = list(INPUT_DIR.rglob("*.json"))
    print(f"📂 Trovati {len(all_files)} file L0 nella cartella di origine.")

    if not all_files:
        print("❌ ERRORE: Nessun file .json trovato in output_late_chunking!")
        return

    manuals_data = {}
    for f in all_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            data["_filename"] = f.name # Memorizziamo il nome file originale temporaneamente
            
            # Usiamo 'source' per raggrupparli (se manca usa 'Manuale_Ignoto')
            src = data.get("source", "Manuale_Ignoto")
            if src not in manuals_data: 
                manuals_data[src] = []
            manuals_data[src].append(data)

    print(f"📚 Identificati {len(manuals_data)} manuali (gruppi semantici) in base ai metadati.")

    # Inizia l'elaborazione per ogni gruppo (Manuale)
    for manual_name, l0_nodes in manuals_data.items():
        print(f"\n⚙️ INIZIO ELABORAZIONE MANUALE: {manual_name}")
        print(f"   🌱 Elaborazione di {len(l0_nodes)} chunk (L0).")
        
        # 3. Gestione dei livelli L1 (Capitoli) - Nessun Limite
        valid_l0 =[n for n in l0_nodes if n.get('vector')]
        vectors = np.array([n['vector'] for n in valid_l0])
        n_samples = len(vectors)
        
        if n_samples == 0:
            print(f"   ⚠️ Nessun embedding trovato per i chunk di {manual_name}. Salto.")
            continue
            
        # Nessun limite: creiamo cluster dinamicamente (es. ~5 chunk per capitolo)
        n_clusters = max(1, int(np.ceil(n_samples / 5.0)))
        
        if n_samples > n_clusters and n_samples > 1:
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42)
            labels = gmm.fit_predict(vectors)
        else:
            labels = [0] * n_samples

        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters: clusters[label] = []
            clusters[label].append(valid_l0[idx])

        print(f"   🧩 Creati dinamicamente {len(clusters)} capitoli (L1).")

        l1_nodes =[]
        
        # 4. Creazione degli L1 e Collegamento L0 -> L1
        for label, group in clusters.items():
            l1_id = f"L1_{l1_counter:04d}"
            l1_counter += 1
            
            # Torno ai chunk (L0) e assegno l'ID del loro padre L1
            for n in group:
                n["L1ID"] = l1_id
            
            # Elaborazione L1: Riassunto + Embedding
            print(f"   🌿 Generazione {l1_id}...")
            summary_text = get_summary([n['text'] for n in group])
            l1_vec = get_embedding(summary_text)
            
            l1_node = {
                "id": l1_id,
                "text": summary_text,
                "vector": l1_vec,
                "level": 1,
                "source": manual_name,
                "child_ids": [n.get('id') for n in group]
            }
            l1_nodes.append(l1_node)

        # 5. Gestione dei livelli L2 (Radice del manuale)
        l2_id = f"L2_{l2_counter:04d}"
        l2_counter += 1
        print(f"   🌳 Generazione Livello Superiore: {l2_id}...")

        # Torno agli L1 e assegno l'ID del loro padre L2
        for l1 in l1_nodes:
            l1["L2ID"] = l2_id

        # Creazione testo riassuntivo L2
        root_text = get_summary([n['text'] for n in l1_nodes])

        # 6. Embedding dei L2: Mean Pooling puro (nessuna API LLM usata qui per il vettore)
        valid_l1_vecs = [n['vector'] for n in l1_nodes if n.get('vector')]
        if valid_l1_vecs:
            l2_vec = np.mean(valid_l1_vecs, axis=0).tolist()
            print("      📊 Embedding L2 calcolato con puro Mean Pooling degli L1.")
        else:
            l2_vec =[] # Sicurezza

        l2_node = {
            "id": l2_id,
            "text": root_text,
            "vector": l2_vec,
            "level": 2,
            "source": manual_name,
            "child_ids": [n['id'] for n in l1_nodes]
        }

        # 7. OUTPUT FINALE -> Tutto salvato "flat" in output_raptor
        print(f"   💾 Salvataggio dei file per {manual_name} in output_raptor...")
        
        # Salva L0 aggiornati (rimuovendo '_filename' prima di salvarli)
        for n in l0_nodes:
            original_filename = n.pop("_filename", f"{n.get('id', time.time())}.json")
            with open(OUTPUT_DIR / original_filename, "w", encoding="utf-8") as f:
                json.dump(n, f, ensure_ascii=False, indent=4)

        # Salva gli L1 appena creati
        for l1 in l1_nodes:
            with open(OUTPUT_DIR / f"{l1['id']}.json", "w", encoding="utf-8") as f:
                json.dump(l1, f, ensure_ascii=False, indent=4)

        # Salva l'L2 appena creato
        with open(OUTPUT_DIR / f"{l2_id}.json", "w", encoding="utf-8") as f:
            json.dump(l2_node, f, ensure_ascii=False, indent=4)

    print(f"\n🏆 PROCEDURA COMPLETATA CON SUCCESSO! Tutti i file (L0 aggiornati, L1 e L2) sono salvati in:\n{OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    run_raptor_step_by_step()