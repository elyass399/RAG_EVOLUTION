import os, json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# --- 1. SETUP AMBIENTE ---
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

# --- 2. CONFIGURAZIONE ---
QDRANT_URL = "http://localhost:6333"
# Suggerisco un nome diverso per distinguerla dalla vecchia KB piatta
COLLECTION_NAME = "zucchetti_raptor_kb" 
VECTOR_SIZE = 768 # Modifica se usi un modello diverso (es. 1024 o 1536)
INPUT_DIR = Path("output_raptor").absolute()

# Inizializza il client Qdrant
client = QdrantClient(url=QDRANT_URL)

# --- 3. MOTORE DI INGESTION RAPTOR ---
def run_raptor_ingestion():
    print(f"\n🧨 Reset database: eliminazione collezione '{COLLECTION_NAME}'...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass # La collezione non esiste ancora
    
    print("🏗️ Creazione nuova collezione...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("✅ Database Qdrant pronto.\n")

    if not INPUT_DIR.exists():
        print(f"❌ ERRORE: La cartella {INPUT_DIR} non esiste!")
        return

    # Troviamo tutti i file JSON (che contengono L0, L1, L2)
    files = list(INPUT_DIR.glob("*.json"))
    print(f"📂 Trovati {len(files)} nodi RAPTOR pronti per l'ingestion.")

    global_id = 1
    points_to_upsert =[]

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Errore lettura JSON: {file_path.name}")
                continue
        
        # 1. Estrazione del vettore (già pre-calcolato!)
        vector = data.get("vector")
        if not vector or not isinstance(vector, list):
            print(f"   ❌ Vettore mancante o non valido in {file_path.name}, salto...")
            continue

        # 2. Costruzione del Payload
        # Salviamo l'ID testuale (es. "L1_0001") nel payload perché Qdrant richiede ID numerici interi o UUID
        payload = {
            "node_id": data.get("id", "ignoto"),
            "text": data.get("text", ""),
            "level": data.get("level", 0), # 0 = Base, 1 = Capitoli, 2 = Radice
            "source": data.get("source", "Ignoto"),
            
            # Relazioni Gerarchiche
            "L1ID": data.get("L1ID", None),
            "L2ID": data.get("L2ID", None),
            "child_ids": data.get("child_ids",[])
        }

        # Aggiungiamo il punto alla lista
        points_to_upsert.append(
            PointStruct(
                id=global_id,
                vector=vector,
                payload=payload
            )
        )
        
        print(f"   [{global_id}] Preparato Nodo Livello {payload['level']} | ID: {payload['node_id']}")
        global_id += 1

    # 3. Caricamento su Qdrant (in batch per massima velocità)
    if points_to_upsert:
        print("\n🚀 Inizio caricamento su Qdrant...")
        # L'upsert accetta una lista di punti. Caricarli tutti in una volta è molto più efficiente.
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert
        )
        print(f"✅ INGESTION COMPLETATA! Totale punti caricati: {len(points_to_upsert)}")
    else:
        print("\n⚠️ Nessun punto valido da caricare.")

if __name__ == "__main__":
    run_raptor_ingestion()