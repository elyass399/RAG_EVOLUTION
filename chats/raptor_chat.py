import os, json, requests, re # Importa librerie di sistema, JSON, richieste web ed espressioni regolari per pulire l'output.
import numpy as np # Importa NumPy per operazioni matematiche e vettoriali.
from dotenv import load_dotenv # Importa il caricatore delle variabili d'ambiente.
from qdrant_client import QdrantClient # Importa il client del database vettoriale Qdrant.
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny # Importa i moduli per creare filtri di ricerca su Qdrant.
from rank_bm25 import BM25Okapi # Importa l'algoritmo di ricerca testuale BM25.

# --- 1. CONFIGURAZIONE E RETE ---
load_dotenv() # Carica in memoria le variabili scritte nel file .env.

user = os.getenv('PROXY_USER', '').replace('@', '%40') # Recupera l'utente proxy e converte la @ in formato sicuro per URL (%40).
password = os.getenv('PROXY_PASS', '') # Recupera la password del proxy aziendale.
host = os.getenv('PROXY_HOST', '') # Recupera l'IP e la porta del proxy (es. 172.16.2.200:8080).
proxy_string = f"http://{user}:{password}@{host}" # Assembla la stringa completa per l'autenticazione.
PROXIES = {"http": proxy_string, "https": proxy_string} # Crea il dizionario da passare a 'requests' per usare il proxy.

QDRANT_URL = "http://localhost:6333" # Definisce l'indirizzo locale del database Qdrant.
COLLECTION_NAME = "zucchetti_raptor_kb" # Nome della collezione che contiene l'albero RAPTOR (L0, L1, L2).
BASE_URL = os.getenv("LITELLM_BASE_URL", "").rstrip("/") # Recupera l'URL base di LiteLLM e toglie lo slash finale.
API_KEY = os.getenv("LITELLM_API_KEY") # Recupera la chiave API dal file .env.
HEADERS = { # Configura le intestazioni della richiesta HTTP.
    "Authorization": f"Bearer {API_KEY}", # Inserisce il token di sicurezza.
    "Content-Type": "application/json", # Indica al server che stiamo parlando in JSON.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36" # Simula di essere un browser Google Chrome per aggirare i blocchi del firewall.
}
client = QdrantClient(url=QDRANT_URL) # Avvia la connessione stabile con Qdrant.

# --- 2. COMUNICAZIONE ROBUSTA ---
def call_api(endpoint, payload, timeout=60): # Crea una funzione centralizzata per comunicare col server LiteLLM.
    url = f"{BASE_URL}/{endpoint}" # Compone l'URL esatto (es. /v1/embeddings).
    try: # Prova a eseguire la richiesta gestendo eventuali errori di rete.
        resp = requests.post(url, json=payload, headers=HEADERS, proxies=PROXIES, timeout=timeout, verify=False) # Invia i dati tramite proxy, ignorando controlli su certificati SSL aziendali.
        if resp.status_code == 200: # Se il server risponde con successo...
            return resp.json() # ...ritorna il risultato convertito in dizionario Python.
    except Exception: # Se si verifica un timeout o una disconnessione...
        pass # ...ignora l'errore per non far crashare lo script.
    return None # Restituisce None in caso di fallimento.

def call_llm(context, question): # Definisce la funzione per interrogare il modello generativo.
    payload = { # Prepara i dati per l'API di chat.
        "model": "gemma3:4b", # Usa il modello veloce per evitare che il proxy tagli la connessione per timeout.
        "messages":[ # Costruisce la conversazione.
            {"role": "system", "content": "Sei l'assistente Zucchetti. Rispondi usando solo il contesto fornito."}, # Il prompt di sistema rigoroso.
            {"role": "user", "content": f"CONTESTO:\n{context}\n\nDOMANDA: {question}"} # Inserisce i testi di Qdrant e la domanda dell'utente.
        ],
        "temperature": 0.1 # Abbassa la creatività per ottenere risposte aderenti ai manuali.
    }
    data = call_api("chat/completions", payload, timeout=120) # Chiama l'API con un timeout di 2 minuti.
    if data and 'choices' in data: # Se la risposta è valida...
        return re.sub(r'</?end_of_turn>', '', data['choices'][0]['message']['content']).strip() # ...pulisce i tag XML strani generati da Gemma e restituisce il testo pulito.
    return "errore di rete o timeout" # Se fallisce, restituisce un avviso all'utente.

# --- 3. LOGICA RAG RAPTOR (BOTTOM-UP) ---
def search_top_l0(query_vec, query_text): # Definisce la funzione per trovare i dettagli L0 più rilevanti analizzando l'intero database.
    all_docs = [] # Inizializza una lista vuota per accumulare tutti i frammenti di livello 0 recuperati.
    next_offset = None # Prepara il segnalibro per gestire la lettura di più pagine di dati da Qdrant.

    while True: # Avvia un ciclo per scaricare i documenti a blocchi finché non sono stati letti tutti.
        res, next_offset = client.scroll(COLLECTION_NAME, scroll_filter=Filter(must=[FieldCondition(key="level", match=MatchValue(value=0))]), limit=500, offset=next_offset, with_payload=True) # Scarica una pagina da 500 documenti L0, usando l'offset per sapere da dove riprendere la lettura.
        all_docs.extend(res) # Aggiunge i documenti appena scaricati alla lista globale in memoria RAM.
        if next_offset is None: break # Interrompe il ciclo di scaricamento quando Qdrant segnala che non ci sono più documenti da leggere.

    if not all_docs: return [] # Restituisce una lista vuota se il database non contiene documenti che soddisfano il filtro.
    
    bm25 = BM25Okapi([d.payload.get('text', '').lower().split() for d in all_docs]) # Inizializza l'algoritmo di ricerca testuale BM25 su tutti i documenti caricati in memoria.
    scores = bm25.get_scores(query_text.lower().split()) # Calcola il punteggio di rilevanza per ogni documento confrontando le parole esatte della domanda.
    
    combined = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True) # Unisce i punteggi ai documenti e crea una classifica ordinata dal risultato migliore al peggiore.
    return [d for s, d in combined[:3]] # Estrae e restituisce i primi 3 documenti della classifica per fornire all'IA il dettaglio più preciso.

def get_parents(l0_nodes): # Funzione fondamentale di RAPTOR: recupera i riassunti (padri) dei dettagli trovati.
    p_ids = {n.payload.get(k) for n in l0_nodes for k in ['L1ID', 'L2ID'] if n.payload.get(k)} # Crea un set (per evitare duplicati) estraendo gli ID dei capitoli (L1) e dei manuali (L2) a cui appartengono i chunk L0 trovati.
    if not p_ids: return[] # Se non ci sono padri, restituisce una lista vuota.
    return client.scroll(COLLECTION_NAME, scroll_filter=Filter(must=[FieldCondition(key="node_id", match=MatchAny(any=list(p_ids)))]), limit=10, with_payload=True)[0] # Interroga Qdrant chiedendo esattamente i nodi che hanno quegli specifici ID (recupero mirato).

# --- 4. CHAT LOOP ---
def start_chat(): # Funzione principale che gestisce il terminale interattivo.
    print("\n🚀 ZUCCHETTI RAG RAPTOR (Gerarchia L0-L1-L2)") # Stampa un messaggio di avvio chiaro.
    while True: # Inizia il ciclo infinito di domande e risposte.
        q = input("\ndomanda : ").strip() # Mostra il prompt pulito, attende l'input e rimuove gli spazi.
        if q.lower() in ["esci", "quit"]: break # Se l'utente digita parole d'uscita, chiude il programma.
        if not q: continue # Se l'utente preme solo invio, ignora e riparte.
        
        emb_data = call_api("embeddings", {"model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"), "input": q}, timeout=30) # Vettorializza la domanda dell'utente chiamando l'API.
        if not emb_data or 'data' not in emb_data: # Se il server embedding non risponde o il proxy blocca...
            print("risposta ai : Errore di rete (Embedding non riuscito).") # ...stampa l'errore senza crashare.
            continue # ...e aspetta una nuova domanda.
            
        vec = emb_data['data'][0]['embedding'] # Estrae il vettore numerico dal JSON.
        l0 = search_top_l0(vec, q) # Passa il vettore e il testo alla ricerca dei dettagli (L0).
        
        if l0: # Se ha trovato almeno un dettaglio pertinente...
            nodes = sorted(l0 + get_parents(l0), key=lambda x: x.payload.get('level', 0), reverse=True) # ...unisce i dettagli (L0) ai loro padri (L1, L2) e li ordina dall'alto verso il basso (L2 -> L1 -> L0).
            context = "\n\n".join([f"---[ Livello {n.payload.get('level')} ]---\n{n.payload.get('text')}" for n in nodes[:4]]) # Crea il super-contesto testuale, prendendo al massimo 4 nodi (es. 1 L2, 1 L1, 2 L0) per non saturare la memoria.
            
            answer = call_llm(context, q) # Passa il super-contesto e la domanda all'LLM.
            print(f"risposta ai : {answer}") # Stampa a schermo la risposta generata dal modello.
        else: # Se la ricerca non trova nessun dettaglio attinente...
            print("risposta ai : info non esiste") # ...risponde immediatamente per risparmiare tempo e risorse di calcolo.

if __name__ == "__main__": # Controlla se il file è eseguito direttamente dal terminale.
    import urllib3 # Importa la libreria per la gestione HTTP a basso livello.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # Silenzia gli avvisi di Python relativi ai certificati SSL non verificati.
    start_chat() # Lancia la chat.