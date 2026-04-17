import os, json, requests, re, time # Importa le librerie standard per interagire con il sistema, richieste web, regex e misurazione del tempo.
import numpy as np # Importa NumPy per la gestione dell'array vettoriale.
from dotenv import load_dotenv # Importa la funzione per caricare le variabili dal file .env.
from qdrant_client import QdrantClient # Importa il client per il database vettoriale Qdrant.

# --- SETUP CONFIGURAZIONE E RETE ---
load_dotenv() # Carica le credenziali e i parametri dal file .env.
user = os.getenv('PROXY_USER', '').replace('@', '%40') # Recupera l'utente proxy e formatta la chiocciola in formato URL-safe (%40).
password = os.getenv('PROXY_PASS', '') # Recupera la password del proxy aziendale.
host = os.getenv('PROXY_HOST', '') # Recupera l'indirizzo e la porta del proxy (es. 172...).
proxy_string = f"http://{user}:{password}@{host}" # Costruisce la stringa di connessione proxy completa con le credenziali.
PROXIES = {"http": proxy_string, "https": proxy_string} # Crea il dizionario per forzare 'requests' a passare dal proxy.

BASE_URL = os.getenv("LITELLM_BASE_URL", "").rstrip("/") # Recupera l'URL base di LiteLLM rimuovendo eventuali slash finali.
HEADERS = { # Definisce le intestazioni HTTP standard per le API.
    "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}", # Inserisce la chiave API per l'autenticazione.
    "Content-Type": "application/json", # Specifica che i dati scambiati saranno in formato JSON.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36" # Camuffa lo script Python come se fosse un browser Chrome.
}

client = QdrantClient(url="http://localhost:6333") # Inizializza la connessione al database vettoriale Qdrant locale.
COLLECTION = "parentchild" # Specifica la collezione in cui sono stati caricati i chunk "figli" con i loro "padri".

MODEL_LLM = "gemma3:4b"  # Usa il modello veloce per evitare che il proxy tronchi la connessione per timeout.
MODEL_EMB = os.getenv("EMBEDDING_MODEL", "nomic-embed-text") # Recupera il modello per gli embedding dalle variabili d'ambiente.

# --- CHIAMATE API ROBUSTE ---
def call_api(endpoint, payload, timeout=60): # Definisce una funzione universale per fare richieste HTTP al server LiteLLM.
    url = f"{BASE_URL}/{endpoint}" # Costruisce l'URL completo unendo la base e l'endpoint (es. /embeddings).
    try: # Avvia un blocco try-except per gestire eventuali cadute di rete senza far crashare il programma.
        resp = requests.post(url, json=payload, headers=HEADERS, proxies=PROXIES, timeout=timeout, verify=False) # Invia la richiesta POST passando dal proxy.
        if resp.status_code == 200: # Se il server risponde con successo (codice 200)...
            return resp.json() # ...restituisce il contenuto della risposta formattato come dizionario Python.
    except Exception: # Se c'è un errore di connessione (timeout, proxy non raggiungibile)...
        pass # ...ignora l'errore per gestirlo silenziosamente.
    return None # Se qualcosa va storto, restituisce None.

def get_embedding(text): # Definisce la funzione per calcolare i vettori.
    data = call_api("embeddings", {"model": MODEL_EMB, "input": text}, timeout=30) # Chiama l'API degli embedding con un timeout breve (30s).
    return np.array(data['data'][0]['embedding']).flatten().tolist() if data else None # Estrae il vettore, lo appiattisce e lo converte in lista standard per Qdrant.

# --- CHAT ENGINE (PARENT-CHILD) ---
def start_chat(): # Definisce la funzione che avvia l'interfaccia utente nel terminale.
    print(f"\n" + "="*60) # Stampa una linea decorativa.
    print("🚀 ZUCCHETTI RAG: PRECISION PARENT-CHILD (Single Call)") # Stampa il titolo aggiornato dell'applicazione.
    print("Logica: Retrieval Figli -> Espansione Padri -> Generazione") # Descrive il flusso logico all'utente.
    print("="*60 + "\n") # Stampa la chiusura della decorazione.
    
    while True: # Avvia il ciclo infinito per continuare a fare domande.
        user_input = input("👤 Tu: ").strip() # Cattura la domanda dell'utente e toglie gli spazi vuoti.
        if user_input.lower() in ["esci", "quit"]: break # Se digita "esci", interrompe il ciclo e chiude lo script.
        if not user_input: continue # Se preme solo invio, ignora e richiede l'input.

        start_time = time.time() # Salva il timestamp attuale per calcolare quanto tempo ci mette a rispondere.

        # 1. EMBEDDING
        q_vec = get_embedding(user_input) # Trasforma la domanda dell'utente in numeri.
        if not q_vec: # Se il server non risponde...
            print("🤖 AI: Errore di connessione al server embedding.") # ...avvisa l'utente.
            continue # ...e salta il resto del ciclo.

        # 2. RETRIEVAL (Trova i figli)
        hits = client.query_points(COLLECTION, query=q_vec, limit=5).points # Cerca su Qdrant i 5 chunk "figli" più simili alla domanda.

        if not hits: # Se il database non trova alcun chunk rilevante...
            print("🤖 AI: Nessuna informazione trovata nella Knowledge Base.") # ...comunica l'assenza di dati.
            continue # ...e attende una nuova domanda.

        # 3. RECUPERO DEI PADRI (Deduplicati)
        parents_to_read = {} # Crea un dizionario vuoto per evitare di passare padri doppi.
        for h in hits: # Inizia a ciclare attraverso i 5 figli trovati.
            p_text = h.payload.get('parent_text', '') # Estrae il testo del "padre" nascosto nel payload del figlio.
            source = h.payload.get('source', 'Sconosciuto') # Estrae il nome del file originale.
            if p_text not in parents_to_read: # Se questo testo padre non è ancora stato inserito nel dizionario...
                parents_to_read[p_text] = source # ...lo salva.
        
        context = "\n\n".join([f"--- DOCUMENTO: {src} ---\n{txt}" for txt, src in parents_to_read.items()]) # Unisce tutti i padri trovati in un'unica stringa.

        # 4. GENERAZIONE FINALE
        print("   ⏳ L'IA sta leggendo i documenti padre espansi...") # Avvisa l'utente dell'inizio generazione.
        sys_prompt = "Sei un analista tecnico. Rispondi alla domanda usando SOLO il contesto fornito. Sii molto dettagliato." # Regole di base per l'LLM.
        
        payload = { # Prepara i dati da inviare a LiteLLM.
            "model": MODEL_LLM, # Inserisce il modello veloce.
            "messages":[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"CONTESTO:\n{context}\n\nDOMANDA: {user_input}"}], # Compone la conversazione.
            "temperature": 0.1 # Abbassa la creatività per avere risposte precise.
        }

        res_llm = call_api("chat/completions", payload, timeout=120) # Invia la richiesta all'LLM.

        if res_llm and 'choices' in res_llm: # Se la chiamata ha successo...
            answer = re.sub(r'</?end_of_turn>', '', res_llm['choices'][0]['message']['content']).strip() # Estrae e pulisce la risposta.
            print(f"\n🤖 AI:\n{answer}") # Stampa a schermo la risposta finale.
            print(f"\n⏱️ Tempo totale: {time.time() - start_time:.2f}s") # Calcola e stampa i secondi impiegati.
        else: # Se c'è stato un timeout o un errore...
            print("\n🤖 AI: Errore di rete o timeout durante la generazione.") # ...stampa l'errore.

if __name__ == "__main__": # Controlla se il file viene eseguito direttamente.
   
    start_chat() # Avvia il programma.