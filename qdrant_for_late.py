import json
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- CONFIGURAZIONE ---
COLLECTION_NAME = "manuali_late_chunking"
QDRANT_URL = "http://localhost:6333"
JSON_DIR = Path("./output_late_chunking")

# Inizializza il client
client = QdrantClient(url=QDRANT_URL)

def upload_data():
    # 1. Verifica se la collezione esiste, altrimenti creala
    # nomic-embed-text ha solitamente 768 dimensioni
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)

    if not exists:
        print(f"🆕 Creazione collezione: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    else:
        print(f"✅ Collezione {COLLECTION_NAME} già esistente.")

    # 2. Preparazione dei punti (Points)
    points = []
    json_files = list(JSON_DIR.glob("*.json"))
    
    print(f"📂 Lettura di {len(json_files)} file JSON...")

    for idx, json_file in enumerate(json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # Creiamo il punto per Qdrant
            point = PointStruct(
                id=idx, # Usiamo un indice numerico semplice come ID
                vector=data['vector'],
                payload={
                    "text": data['text'],
                    "context": data['context'],
                    "source": data['source'],
                    "chunk_id": data['id']
                }
            )
            points.append(point)

    # 3. Caricamento massivo (Upsert)
    if points:
        print(f"🚀 Caricamento di {len(points)} punti su Qdrant...")
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        print(f"🏆 Ingestione completata con successo!")
    else:
        print("⚠️ Nessun dato trovato da caricare.")

if __name__ == "__main__":
    upload_data()