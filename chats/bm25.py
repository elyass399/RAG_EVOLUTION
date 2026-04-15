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
LLM_MODEL = "gemma3:27b-it-qat"
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
    payload = {"model": LLM_MODEL, "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}], "temperature": 0}
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

# --- ROUTING MANUALE ---
#Usa LLM per capire a quale manuale appartiene la domanda, migliorando la ricerca.
def get_target_manual(query):   # Definisce la funzione di classificazione che valuta la domanda dell'utente.
    prompt = f"Data la domanda: '{query}', scegli il manuale: 'calcio', 'guerre', 'prompt'. Rispondi solo con la parola."  # Crea un prompt costringendo l'LLM a rispondere unicamente con una di queste 3 parole.
    res = call_llm("Sei un classificatore.", prompt) # Invia il prompt all'LLM assegnandogli il ruolo di classificatore rigoroso.
    if res:   # Verifica che l'LLM abbia generato effettivamente una risposta testuale.
        res = res.lower() # Converte la risposta dell'LLM in minuscolo per fare confronti sicuri.
        for m in ['calcio', 'guerre', 'prompt']: # Inizia un ciclo scorrendo le 3 categorie preimpostate.
            if m in res: return m  # Se la categoria è contenuta nella risposta dell'LLM, restituisce la stringa della categoria.
    return None # Se fallisce, risponde in modo confuso o c'è un errore, restituisce None (la ricerca avverrà su tutto il DB).

# --- RICERCA BM25 + SEMANTIC RERANK ---
def hybrid_bm25_rerank(query_text, query_vec, manual):   # Definisce la funzione che fonde Keyword Search e Semantic Search.
    # 1. Filtro manuale
    q_filter = Filter(must=[FieldCondition(key="manual", match=MatchValue(value=manual))]) if manual else None  # Se il router ha trovato un manuale, crea un filtro Qdrant per pescare solo lì dentro, altrimenti non usa filtri.
    
    # 2. Recupera documenti filtrati
    all_docs = client.scroll(COLLECTION, scroll_filter=q_filter, limit=500, with_payload=True)[0]  # Preleva da Qdrant fino a 500 documenti che superano il filtro, includendo il loro testo (payload).
    if not all_docs: return []  # Se non ci sono documenti disponibili, interrompe la ricerca e restituisce una lista vuota.

    # 3. BM25
    # Creiamo un corpus testuale per BM25 e calcoliamo i punteggi
    corpus = [d.payload['text'] for d in all_docs]  # Crea una lista estraendo il campo 'text' da tutti i 500 documenti recuperati.
    tokenized_corpus = [doc.lower().split() for doc in corpus]  # Trasforma tutto il testo in minuscolo e lo divide in una lista di singole parole (tokenizzazione).
    bm25 = BM25Okapi(tokenized_corpus)   # Inizializza l'algoritmo BM25 passandogli tutti i documenti suddivisi in parole.
    tokenized_query = query_text.lower().split()  # Scompone in parole minuscole anche la domanda originale dell'utente.
    bm25_scores = bm25.get_scores(tokenized_query)  # Calcola quanto bene la query matcha con le parole esatte di ciascun documento (genera una lista di punteggi).

    # 4. Semantic search
    #Ricerca semantica con embedding e salva score.
    semantic_results = client.query_points(COLLECTION, query=query_vec, query_filter=q_filter, limit=20).points   # Effettua la vera interrogazione vettoriale su Qdrant, chiedendo i 20 nodi migliori.
    semantic_scores = {r.id: r.score for r in semantic_results}  # Crea un dizionario rapido in cui la chiave è l'ID del documento e il valore è il suo score semantico (es. 0.85).

    # 5. Rerank combinato
    #Combina BM25 + semantico con pesi e crea lista di tuple (score, doc), poi ordina e prendi top 15.
    combined = []  # Inizializza una lista vuota per raccogliere i documenti con il loro punteggio finale.
    for idx, doc in enumerate(all_docs):  # Passa in rassegna tutti i 500 documenti scaricati all'inizio usando il loro indice numerico (idx).
        doc_id = doc.id  # Estrae l'ID univoco di Qdrant del documento in esame.
        bm25_score = bm25_scores[idx]   # Recupera il punteggio esatto (keyword) calcolato al passaggio 3 per questo documento.
        semantic_score = semantic_scores.get(doc_id, 0)  # Cerca il punteggio semantico nel dizionario; se Qdrant non lo ha ritenuto tra i top 20, assegna uno score di 0.
        final_score = 0.4 * bm25_score + 0.6 * semantic_score  # pesi: semantico più importante# pesi: semantico più importante : # Applica la formula matematica per mescolare il peso della parola esatta (40%) e del concetto (60%).
        combined.append((final_score, doc)) # Salva una tupla contenente il punteggio finale e il documento intero nella lista.

    combined.sort(key=lambda x: x[0], reverse=True)  # Ordina la lista di tuple basandosi sul punteggio (elemento 0 della tupla), dal numero più grande al più piccolo.
    top_docs = [doc for score, doc in combined[:15]]   # Scarta il punteggio e conserva solo gli oggetti "documento" per i migliori 15 risultati in classifica.

    return top_docs  # Ritorna i documenti vincitori da passare al LLM.

# --- CHAT LOOP ---
#Loop principale: prende input utente → espande query → identifica manuale → embedding → ricerca + rerank → genera risposta con LLM.
def start_chat():  # Definisce la funzione che gestisce l'interfaccia a riga di comando.
    print(f"\n🚀 ZUCCHETTI RAG - BM25 + Semantic Rerank")  # Stampa il benvenuto visivo per l'utente.
    while True:  # Avvia il ciclo infinito permettendo di fare domande in sequenza.
        user_input = input("\n👤 Tu: ").strip()   # Cattura il testo scritto dall'utente tramite tastiera e rimuove gli spazi iniziali/finali.
        if user_input.lower() in ["esci", "quit"]: break  # Se la stringa digitata è "esci" o "quit", chiude il ciclo e spegne il programma.
        if not user_input: continue   # Se l'utente preme solo Invio, il programma salta l'esecuzione e attende una nuova digitazione.

        rich_query = expand_query(user_input)  # Trasforma eventuali parole singole in frasi complete tramite l'apposita funzione.
        manual_filter = get_target_manual(rich_query)  # Chiede all'LLM (Router) di provare a indovinare di che manuale stiamo parlando.
        print(f"   🎯 Manuale rilevato: {manual_filter}") # Stampa per l'utente se l'IA ha deciso di filtrare una categoria o se agirà globalmente (None).

        query_vec = get_embedding(rich_query)  # Converte la frase "arricchita" nei vettori numerici usando l'endpoint embeddings.
        results = hybrid_bm25_rerank(user_input, query_vec, manual_filter) # Scatena il motore ibrido passandogli il testo, il vettore e l'eventuale filtro di categoria.

        if results:  # Se la ricerca nel DB ha effettivamente prodotto dei documenti validi...
            context_txt = "\n\n".join([f"--- DOC [ID {i}] ---\n{r.payload['text']}" for i, r in enumerate(results)]) # formatta i documenti unendoli in un enorme blocco di testo, separati graficamente.
            risposta = call_llm(SYSTEM_PROMPT, f"DOMANDA: {user_input}\n\nCONTESTO:\n{context_txt}")  # Passa al modello pesante (LLM) le istruzioni, il testo recuperato e la domanda dell'utente.
            print(f"\n🤖 Gemma3:\n{risposta}") # Stampa finalmente la risposta formulata e articolata dall'IA.
        else:  # Se il motore non ha trovato assolutamente nulla nel database...
            print("🤖 info non esiste") # Non interroga l'LLM pesante e risponde direttamente comunicando che i dati mancano.

if __name__ == "__main__":
    start_chat()