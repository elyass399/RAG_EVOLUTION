import os, subprocess, json, time, re
from pathlib import Path
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi

# --- ENV & PROXY ---
load_dotenv()
os.environ["no_proxy"] = "localhost,127.0.0.1,padova.zucchettitest.it"

# --- CLIENT & MODELLI ---
client = QdrantClient(url="http://localhost:6333")
COLLECTION = "zucchetti_knowledge_base" 
LLM_MODEL = "gemma3:4b"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")

SYSTEM_PROMPT = """Sei l'Assistente RAG Ultra-Preciso. 
Usa i documenti forniti per rispondere. Se trovi dettagli su battaglie, persone o regole, riportali.
Se l'informazione non è presente, rispondi: "info non esiste"."""

# --- POWERSHELL BRIDGE ---
def _ps_invoke(url_suffix, payload):
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL").rstrip('/')
    url = f"{base_url}/{url_suffix}"
    stamp = str(time.time()).replace('.', '')
    req_file = Path(f"req_{stamp}.json").absolute().as_posix()
    res_file = Path(f"res_{stamp}.bin").absolute().as_posix()
    with open(req_file, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False)
    ps_cmd = f'$url="{url}"; $headers=@{{"Authorization"="Bearer {api_key}";"Content-Type"="application/json;charset=utf-8"}}; try{{$body=[System.IO.File]::ReadAllBytes("{req_file}");$res=Invoke-WebRequest -Uri $url -Method Post -Headers $headers -Body $body -Proxy $null -UseBasicParsing -TimeoutSec 60;[System.IO.File]::WriteAllBytes("{res_file}", $res.RawContentStream.ToArray())}} catch{{$errMsg="ERROR: "+$_.Exception.Message;[System.IO.File]::WriteAllBytes("{res_file}", [System.Text.Encoding]::UTF8.GetBytes($errMsg))}}'
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    data = None
    if os.path.exists(res_file):
        with open(res_file, "rb") as f: raw = f.read()
        if not raw.startswith(b"ERROR"):
            try: data = json.loads(raw.decode('utf-8-sig'))
            except: data = None
    if os.path.exists(req_file): os.remove(req_file)
    if os.path.exists(res_file): os.remove(res_file)
    return data

# --- EMBEDDING ---
#Richiama il modello embedding e restituisce un vettore numerico.
def get_embedding(text): # Definisce la funzione per calcolare l'embedding (vettore numerico) di una stringa di testo.
    data = _ps_invoke("embeddings", {"model": EMBED_MODEL, "input": text}) # Invia la richiesta all'endpoint delle API per gli embedding, passando il modello scelto e il testo da vettorializzare.
    return np.array(data['data'][0]['embedding']).flatten().tolist() if data else None # Estrae l'array dal JSON, si assicura che sia monodimensionale (flatten), lo converte in una lista Python standard per Qdrant e, se la chiamata fallisce, restituisce None.

# --- LLM CALL ---
#Richiama il modello LLM con un prompt di sistema e una domanda utente, restituendo la risposta testuale.
def call_llm(sys, usr):  # Definisce la funzione che accetta un prompt di sistema (sys, istruzioni) e l'input dell'utente (usr, la domanda).
    payload = {"model": LLM_MODEL, "messages":[{"role": "system", "content": sys}, {"role": "user", "content": usr}], "temperature": 0}
    data = _ps_invoke("chat/completions", payload) # Costruisce il dizionario JSON per l'API, impostando i ruoli dei messaggi e la temperatura a 0 per avere risposte estremamente fattuali e zero creatività.
    if data and 'choices' in data:  # Verifica che l'API abbia risposto correttamente e che la struttura del JSON contenga le scelte generate.
        return re.sub(r'</?end_of_turn>', '', data['choices'][0]['message']['content']).strip() # Estrae il testo della risposta, usa le espressioni regolari per eliminare eventuali "tag sporchi" lasciati dal modello (come <end_of_turn>) e rimuove gli spazi in eccesso.
    return None  # Se la chiamata è fallita o il JSON è vuoto, restituisce None come sistema di sicurezza.

# --- ESPANSIONE QUERY ---
#Se la query è una sola parola, la arricchisce per aiutare BM25 e LLM.
def expand_query(text): # Definisce la funzione che prende in input il testo digitato dall'utente.
    if len(text.split()) == 1: # Divide il testo in parole usando gli spazi; se risulta esserci una parola sola...
        return f"Informazioni e dettagli storici o tecnici su {text}"  # ...crea e restituisce una frase standard per dare maggiore contesto semantico al motore di ricerca.
    return text # Se la query contiene già più di una parola, la restituisce invariata.

# (LA FUNZIONE ROUTER MANUALE È STATA COMPLETAMENTE RIMOSSA DA QUI)

# --- RICERCA BM25 + SEMANTIC RERANK ---
def hybrid_bm25_rerank(query_text, query_vec):   # Rimosso il parametro 'manual', la ricerca ora è autonoma.
    
    # 1. Recupera documenti senza filtro forzato
    all_docs = client.scroll(COLLECTION, limit=500, with_payload=True)[0]  # Preleva da Qdrant i documenti per l'analisi BM25.
    if not all_docs: return[]  # Se non ci sono documenti disponibili, interrompe la ricerca e restituisce una lista vuota.

    # 2. BM25
    # Creiamo un corpus testuale per BM25 e calcoliamo i punteggi
    corpus =[d.payload['text'] for d in all_docs]  # Crea una lista estraendo il campo 'text' da tutti i 500 documenti recuperati.
    tokenized_corpus =[doc.lower().split() for doc in corpus]  # Trasforma tutto il testo in minuscolo e lo divide in liste di parole.
    bm25 = BM25Okapi(tokenized_corpus)   # Inizializza l'algoritmo BM25 passandogli i documenti.
    tokenized_query = query_text.lower().split()  # Scompone in parole minuscole anche la domanda originale dell'utente.
    bm25_scores = bm25.get_scores(tokenized_query)  # Calcola quanto bene la query matcha con le parole esatte di ciascun documento.

    # 3. Semantic search
    # Ricerca semantica con embedding. Questa operazione "emergerà" automaticamente i documenti del manuale corretto.
    semantic_results = client.query_points(COLLECTION, query=query_vec, limit=20).points   # Effettua l'interrogazione vettoriale pura chiedendo i 20 nodi migliori.
    semantic_scores = {r.id: r.score for r in semantic_results}  # Crea un dizionario rapido (ID -> Score semantico).

    # 4. Rerank combinato
    # Combina BM25 + semantico con pesi e crea lista di tuple (score, doc).
    combined =[]  # Inizializza una lista vuota per raccogliere i documenti con il loro punteggio finale.
    for idx, doc in enumerate(all_docs):  # Passa in rassegna tutti i documenti.
        doc_id = doc.id  # Estrae l'ID univoco.
        bm25_score = bm25_scores[idx]   # Recupera il punteggio esatto keyword.
        semantic_score = semantic_scores.get(doc_id, 0)  # Cerca il punteggio semantico nel dizionario.
        final_score = 0.4 * bm25_score + 0.6 * semantic_score  # pesi: semantico più importante (60%).
        combined.append((final_score, doc)) # Salva una tupla contenente il punteggio finale e il documento.

    combined.sort(key=lambda x: x[0], reverse=True)  # Ordina la lista basandosi sul punteggio, dal più grande al più piccolo.
    top_docs = [doc for score, doc in combined[:15]]   # Conserva solo gli oggetti "documento" per i migliori 15 risultati.

    return top_docs  # Ritorna i documenti vincitori da passare al LLM.

# --- CHAT LOOP ---
# Loop principale: prende input utente → espande query → embedding → ricerca + rerank → genera risposta con LLM.
def start_chat():  # Definisce la funzione che gestisce l'interfaccia a riga di comando.
    print(f"\n🚀 ZUCCHETTI RAG - BM25 + Semantic Rerank (Single Call Ottimizzata)")  # Stampa il benvenuto visivo per l'utente.
    while True:  # Avvia il ciclo infinito permettendo di fare domande in sequenza.
        user_input = input("\n👤 Tu: ").strip()   # Cattura il testo scritto dall'utente tramite tastiera.
        if user_input.lower() in ["esci", "quit"]: break  # Se la stringa digitata è "esci" o "quit", chiude il ciclo.
        if not user_input: continue   # Se l'utente preme solo Invio, salta l'esecuzione.

        rich_query = expand_query(user_input)  # Trasforma eventuali parole singole in frasi complete tramite l'apposita funzione.

        # Nessun router manuale qui! Passiamo dritti all'embedding e facciamo fare il lavoro alla matematica.
        query_vec = get_embedding(rich_query)  # Converte la frase "arricchita" nei vettori numerici.
        results = hybrid_bm25_rerank(user_input, query_vec) # Scatena il motore ibrido passandogli SOLO il testo e il vettore.

        if results:  # Se la ricerca nel DB ha effettivamente prodotto dei documenti validi...
            context_txt = "\n\n".join([f"--- DOC [ID {i}] ---\n{r.payload['text']}" for i, r in enumerate(results)]) # formatta i documenti in un blocco di testo.
            risposta = call_llm(SYSTEM_PROMPT, f"DOMANDA: {user_input}\n\nCONTESTO:\n{context_txt}")  # Passa all'LLM le istruzioni, il testo recuperato e la domanda dell'utente.
            print(f"\n🤖 Gemma3:\n{risposta}") # Stampa finalmente la risposta formulata dall'IA.
        else:  # Se il motore non ha trovato assolutamente nulla nel database...
            print("🤖 info non esiste") # Non interroga l'LLM pesante e risponde direttamente.

if __name__ == "__main__":
    start_chat()