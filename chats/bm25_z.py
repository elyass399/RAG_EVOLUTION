import os, subprocess, json, time, re
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi

# --- ENV ---
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "manuali_late_chunking"
LLM_MODEL = "gemma3:27b-it-qat"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")

SYSTEM_PROMPT = """Sei un assistente RAG ultra-preciso.
Rispondi SOLO usando il contesto fornito.
Se l'informazione non è presente: "info non esiste".
Cita il numero del documento quando possibile."""

# --- POWERSHELL ---
def _ps_invoke(url_suffix, payload):
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL").rstrip('/')
    url = f"{base_url}/{url_suffix}"
    stamp = str(time.time()).replace('.', '')
    req_file = Path(f"req_{stamp}.json").absolute().as_posix()
    res_file = Path(f"res_{stamp}.bin").absolute().as_posix()

    with open(req_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    ps_cmd = f'$url="{url}"; $headers=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json;charset=utf-8"}}; try{{$body=[System.IO.File]::ReadAllBytes("{req_file}");$res=Invoke-WebRequest -Uri $url -Method Post -Headers $headers -Body $body -Proxy $null -UseBasicParsing -TimeoutSec 60;[System.IO.File]::WriteAllBytes("{res_file}", $res.RawContentStream.ToArray())}} catch{{$errMsg="ERROR: "+$_.Exception.Message;[System.IO.File]::WriteAllBytes("{res_file}", [System.Text.Encoding]::UTF8.GetBytes($errMsg))}}'

    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)

    if not os.path.exists(res_file):
        return None

    with open(res_file, "rb") as f:
        raw = f.read()

    os.remove(req_file)
    os.remove(res_file)

    if raw.startswith(b"ERROR"):
        return None

    try:
        return json.loads(raw.decode('utf-8-sig'))
    except:
        return None


# --- EMBEDDING ---
def get_embedding(text): # Definisce la funzione per trasformare il testo in numeri.
    data = _ps_invoke("embeddings", {"model": EMBED_MODEL, "input": text}) # Chiama il server tramite PowerShell per chiedere l'embedding.
    if not data: # Se la chiamata fallisce...
        return None  # ...ritorna None.
    return np.array(data['data'][0]['embedding']).flatten().tolist() # Estrae il vettore, lo appiattisce e lo converte in una lista Python standard per Qdrant.


# --- LLM ---
def call_llm(sys, usr):  # Definisce la funzione per parlare con Gemma/LLM.
    payload = { # Crea il dizionario con i dati da inviare.
        "model": LLM_MODEL, # Specifica il modello da usare.
        "messages": [ # Crea la cronologia della conversazione.
            {"role": "system", "content": sys}, # Inserisce il prompt di sistema (le regole).
            {"role": "user", "content": usr} # Inserisce la domanda o il prompt dell'utente.
        ],
        "temperature": 0
    }
    data = _ps_invoke("chat/completions", payload) # Invia il payload al server tramite PowerShell.
    if data and 'choices' in data: # Verifica che la risposta sia valida e contenga il testo.
        return re.sub(r'</?end_of_turn>', '', data['choices'][0]['message']['content']).strip() # Estrae il testo, rimuove eventuali tag xml spuri generati da Gemma e toglie spazi inutili.
    return "errore LLM" # Se qualcosa va storto, ritorna un messaggio di errore leggibile.


# --- QUERY EXPANSION ---
def expand_query(text):  # Definisce la funzione per allungare le query troppo brevi.
    if len(text.split()) <= 2: # Se l'utente digita solo 1 o 2 parole...
        return f"Spiegazione dettagliata e informazioni su {text}" # ...aggiunge contesto per aiutare sia la ricerca semantica che il BM25 a trovare più match.
    return text   # Altrimenti restituisce la query originale.


# --- CLASSIFICATION (SOFT USE) ---
def get_target_manual(query): # Definisce la funzione che capisce di che argomento stiamo parlando.
    prompt = f"Domanda: '{query}'. Categoria: calcio, guerre, prompt. Rispondi solo con una parola." # Costringe il modello a scegliere una singola categoria.
    res = call_llm("Sei un classificatore.", prompt) # Invia la richiesta al modello usando un system prompt brevissimo.
    if res: # Se il modello risponde...
        res = res.lower() # ...converte in minuscolo.
        for m in ['calcio', 'guerre', 'prompt']: # Controlla le tre categorie valide.
            if m in res: # Se la categoria è presente nella risposta del modello...
                return m  # ...restituisce il nome della categoria (il filtro).
    return None  # Ritorna None se il modello fa confusione o la domanda non c'entra nulla.


# --- NORMALIZATION (FIXED Bug) ---
def normalize(scores):  # Definisce la funzione che porta i punteggi su una scala da 0.0 a 1.0.
    if scores is None or len(scores) == 0:  # Se la lista è vuota o inesistente...
        return scores  # ...la restituisce così com'è.

    scores = np.array(scores)  # Converte la lista in un array NumPy per fare matematica veloce.

    min_s = np.min(scores) # Trova il punteggio più basso.
    max_s = np.max(scores)  # Trova il punteggio più alto.

    if max_s - min_s == 0: # Se tutti i punteggi sono identici (evita la divisione per zero)...
        return [0.5] * len(scores)   # ...assegna a tutti un valore medio di 0.5.

    return ((scores - min_s) / (max_s - min_s)).tolist()  # Applica la formula di normalizzazione Min-Max e ritrasforma in lista Python.


# --- HYBRID SEARCH ---
def hybrid_search(query_text, query_vec, manual):  # Definisce la ricerca che unisce vettori (significato) e BM25 (parole esatte).


    # 1. Vector search first (reduce scope)
    semantic_results = client.query_points( # Esegue la query su Qdrant.
        COLLECTION,  # Specifica la collezione.
        query=query_vec,  # Cerca usando l'embedding della domanda dell'utente.
        limit=100  # Prende i 100 documenti semanticamente più simili (non applica il filtro manuale qui).
    ).points

    if not semantic_results: # Se non trova nulla...
        return [] # ...ritorna lista vuota.

    docs = []   # Lista per conservare gli oggetti Qdrant originali.
    texts = [] # Lista per conservare solo il testo dei documenti.
    semantic_scores = []  # Lista per conservare il punteggio semantico (coseno) di ogni documento.

    for r in semantic_results: # Cicla sui 100 risultati ottenuti da Qdrant.
        docs.append(r) # Salva l'oggetto intero.
        texts.append(r.payload.get("text", "")) # Estrae il campo di testo per il BM25.
        semantic_scores.append(r.score)  # Salva il punteggio vettoriale.

    # 2. BM25 on reduced set
    tokenized_corpus = [t.lower().split() for t in texts] # Trasforma i testi in liste di parole minuscole.
    bm25 = BM25Okapi(tokenized_corpus) # Crea l'indice BM25 su quei 100 documenti.
    bm25_scores = bm25.get_scores(query_text.lower().split())  # Calcola il punteggio di corrispondenza esatta delle parole per ogni documento.

    # 3. Normalize
    bm25_norm = normalize(bm25_scores) # Normalizza (0-1) i punteggi delle parole chiave.
    semantic_norm = normalize(semantic_scores) # Normalizza (0-1) i punteggi dei vettori.

    # 4. Combine scores + soft boost
    combined = []  # Inizializza la lista finale.

    for i, doc in enumerate(docs): # Cicla i 100 documenti con il loro indice numerico.
        score = 0.6 * semantic_norm[i] + 0.4 * bm25_norm[i]  # Unisce i punteggi dando più importanza al significato (60%) che alla parola esatta (40%).

        # Soft boost if manual matches
        if manual and doc.payload.get("manual") == manual:  # Se il router aveva predetto un manuale e il documento appartiene proprio a quello...
            score += 0.15 # ...aggiunge un bonus del 15% al punteggio finale del documento, spingendolo in alto nella classifica.


        combined.append((score, doc))  # Salva la tupla (punteggio, documento) nella lista.

    combined.sort(key=lambda x: x[0], reverse=True) # Ordina la lista in base al punteggio, dal più alto al più basso.

    return [doc for score, doc in combined[:8]]  # limit context : # Ritorna solo i primi 8 documenti classificati, rimuovendo il punteggio dalla lista restituita.


# --- CHAT LOOP ---
def start_chat(): # Definisce la funzione principale che avvia il programma.
    print("\n🚀 RAG OPTIMIZED VERSION") # Stampa un messaggio di avvio.

    while True:  # Inizia il loop infinito per la chat.
        user_input = input("\n👤 Tu: ").strip()  # Chiede l'input all'utente e rimuove gli spazi vuoti agli estremi.

        if user_input.lower() in ["esci", "quit"]: # Se l'utente vuole uscire...
            break # ...rompe il loop e chiude il programma.

        if not user_input:  # Se l'utente non ha scritto nulla e preme Invio...
            continue  # ...ricomincia il ciclo chiedendo un nuovo input.

        query = expand_query(user_input)  # Applica l'espansione della query (se è troppo corta).
        manual = get_target_manual(query)  # Chiama l'LLM per classificare la domanda in un manuale.

        print(f"🎯 Categoria stimata: {manual}")

        query_vec = get_embedding(query)  # Converte la query (espansa) in vettori.

        if not query_vec:  # Se la chiamata per l'embedding fallisce...
            print("Errore embedding")  # ...avvisa l'utente.
            continue  # ...e riparte con una nuova domanda.

        # ⚠️ IMPORTANT FIX: use expanded query
        results = hybrid_search(query, query_vec, manual)  # Avvia la ricerca ibrida nel database usando vettori, testo e la spinta del manuale.

        if not results:  # Se non viene trovato alcun documento rilevante...
            print("🤖 info non esiste")
            continue

        # Build clean context
        context_blocks = []
        for i, r in enumerate(results):
            text = r.payload.get("text", "")
            context_blocks.append(f"Documento {i+1}:\n{text}")

        context = "\n\n".join(context_blocks)

        prompt = f"""
DOMANDA:
{user_input}

CONTESTO:
{context}
"""

        answer = call_llm(SYSTEM_PROMPT, prompt)
        print(f"\n🤖 Risposta:\n{answer}")


if __name__ == "__main__":
    start_chat()