import os, json, requests, re, time # Importa le librerie per gestire sistema, JSON, richieste web, espressioni regolari e tempo.
from dotenv import load_dotenv # Importa la funzione per leggere il file .env con le credenziali aziendali.
from falkordb import FalkorDB # Importa il client ufficiale per interagire con il database a grafo FalkorDB.

load_dotenv() # Carica le variabili d'ambiente come API_KEY e URL dal file .env nella memoria locale.
user = os.getenv('PROXY_USER', '').replace('@', '%40') # Recupera l'utente proxy e trasforma la chiocciola in formato URL-safe.
password = os.getenv('PROXY_PASS', '') # Legge la password del proxy aziendale dalle impostazioni di sistema.
host = os.getenv('PROXY_HOST', '') # Recupera l'indirizzo IP e la porta del proxy (es. 172.16.2.200:8080).
proxy_string = f"http://{user}:{password}@{host}" # Costruisce la stringa di connessione completa per l'autenticazione al proxy.
PROXIES = {"http": proxy_string, "https": proxy_string} # Configura il dizionario dei proxy per forzare le richieste web attraverso il gateway.

BASE_URL = os.getenv("LITELLM_BASE_URL", "").rstrip("/") # Prende l'URL base del server LiteLLM rimuovendo eventuali barre finali.
HEADERS = {"Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"} # Imposta le intestazioni di sicurezza e simula un browser per evitare blocchi firewall.

db = FalkorDB(host='localhost', port=6380) # Stabilisce la connessione con l'istanza locale di FalkorDB sulla porta 6380.
graph = db.select_graph('ZucchettiPureGraph') # Seleziona il grafo specifico contenente la conoscenza Zucchetti.
LLM_MODEL = "gemma3:4b" # Imposta il modello linguistico veloce per garantire risposte rapide sotto i limiti del proxy.

def call_api(endpoint, payload, timeout=120): # Definisce la funzione per inviare richieste POST sicure al server LiteLLM.
    url = f"{BASE_URL}/{endpoint}" # Compone l'indirizzo completo per l'endpoint specifico (es. chat/completions).
    try: # Avvia un blocco di protezione per gestire eventuali errori di rete o timeout.
        resp = requests.post(url, json=payload, headers=HEADERS, proxies=PROXIES, timeout=timeout, verify=False) # Invia i dati al server passando dal proxy e ignorando certificati SSL non validi.
        if resp.status_code == 200: return resp.json() # Se la risposta è positiva, restituisce il contenuto JSON decodificato.
    except Exception: pass # In caso di errore di connessione, lo ignora per non interrompere l'esecuzione dello script.
    return None # Restituisce None se la chiamata non è andata a buon fine.

def get_graph_context(query): # Definisce la funzione principale per estrarre il contesto logico dal database a grafo.
    extracted_words = re.findall(r'\b[A-Z0-9]{2,}\b|\b[a-zA-Z]{4,}\b', query) # Estrae acronimi tecnici (SAP, RAG) o parole comuni lunghe, distinguendo tra maiuscole e minuscole.
    raw_words = [w.upper() for w in extracted_words] # Converte tutte le parole trovate in maiuscolo per uniformarle ai dati del grafo.
    ignore = ["DOMANDA", "QUESTO", "QUALE", "QUANDO", "SISTEMA", "INFORMAZIONI", "PERIODO", "DURATA"] # Lista di parole generiche da escludere dalla ricerca semantica.
    keywords = [w for w in raw_words if w not in ignore] # Filtra le parole grezze mantenendo solo i concetti chiave reali.

    if not keywords: # Se il filtro ha rimosso tutto (domanda troppo semplice), attiva una ricerca di emergenza.
        keywords = [w.upper() for w in re.findall(r'\w{2,}', query) if w.upper() not in ignore] # Fallback: accetta qualsiasi parola di almeno 2 lettere per trovare un aggancio.

    candidate_chunks = {} # Inizializza un contenitore per calcolare la rilevanza dei paragrafi trovati.
    for kw in keywords: # Cicla su ogni parola chiave identificata nella domanda.
        q = f"MATCH (ch:Chunk) WHERE ch.content CONTAINS '{kw}' RETURN ch.content LIMIT 10" # Crea la query Cypher per trovare i testi che contengono la parola chiave.
        res = graph.query(q) # Invia la query al motore FalkorDB.
        for row in res.result_set: # Esamina ogni riga di testo restituita dal database.
            text = row[0] # Estrae il contenuto del paragrafo.
            if text not in candidate_chunks: candidate_chunks[text] = 0 # Se il chunk è nuovo, lo inserisce nel dizionario con punteggio zero.
            candidate_chunks[text] += 1 # Incrementa il punteggio del chunk basandosi sulla quantità di keyword corrispondenti.

    sorted_chunks = sorted(candidate_chunks.items(), key=lambda x: x[1], reverse=True) # Ordina i paragrafi dal più pertinente al meno pertinente.
    best_chunks = [c[0] for c in sorted_chunks[:5]] # Seleziona i 5 migliori risultati per costruire il contesto dell'LLM.

    facts = [] # Crea una lista per raccogliere le relazioni logiche (fatti) tra i concetti.
    seen_facts = set() # Utilizza un set per evitare di inserire relazioni identiche più volte.
    for text in best_chunks: # Scorre i testi selezionati per trovare legami logici aggiuntivi.
        for kw in keywords: # Ripercorre le parole chiave per interrogare i nodi del grafo.
            q_rel = f"MATCH (c:Concept) WHERE c.name CONTAINS '{kw}' MATCH (c)-[r]->(m:Concept) RETURN c.name, type(r), m.name LIMIT 5" # Query Cypher: trova le relazioni che collegano i concetti chiave.
            res_rel = graph.query(q_rel) # Esegue la ricerca delle relazioni su FalkorDB.
            for row in res_rel.result_set: # Cicla sui legami trovati nel grafo.
                f = f"{row[0]} {row[1].replace('_', ' ')} {row[2]}" # Formatta la relazione in una frase leggibile (Soggetto Relazione Oggetto).
                if f not in seen_facts: # Se il fatto non è un duplicato...
                    facts.append(f) # ...lo aggiunge alla lista dei fatti certi.
                    seen_facts.add(f) # ...e lo registra nel set di controllo.

    return facts[:20], best_chunks # Restituisce le prime 20 relazioni logiche e i 5 paragrafi testuali.

def start_pure_chat(): # Definisce la funzione che gestisce l'interfaccia di chat interattiva.
    print("\n🚀 ZUCCHETTI PURE GRAPHRAG - UNIVERSAL LOGIC MODE") # Stampa l'intestazione di avvio del programma.
    while True: # Avvia un ciclo infinito per permettere una conversazione continua.
        user_input = input("\ndomanda : ").strip() # Attende la domanda dell'utente e rimuove gli spazi bianchi inutili.
        if user_input.lower() in ["esci", "quit"]: break # Chiude lo script se l'utente digita un comando di uscita.
        if not user_input: continue # Ignora gli invii a vuoto e richiede un nuovo input.

        facts, chunks = get_graph_context(user_input) # Interroga il grafo per ottenere sia il testo che le relazioni logiche.
        if not chunks: # Se il database non contiene alcuna informazione utile...
            print("risposta ai : info non esiste") # ...risponde immediatamente evitando allucinazioni.
            continue # Riparte chiedendo una nuova domanda.

        graph_txt = "\n".join(f"- {f}" for f in facts) # Trasforma la lista delle relazioni in un elenco puntato testuale.
        docs_txt = "\n\n".join(chunks) # Unisce i paragrafi trovati in un unico blocco di testo separato.
        sys_prompt = "Sei l'assistente GraphRAG. Rispondi usando CONTESTO e RELAZIONI in linguaggio naturale. Se non c'è info, di' che non lo sai." # Definisce le regole di comportamento per l'IA.
        user_prompt = f"CONTESTO TESTUALE:\n{docs_txt}\n\nRELAZIONI LOGICHE:\n{graph_txt}\n\nDOMANDA: {user_input}" # Crea il prompt finale unendo dati certi e domanda utente.

        payload = {"model": LLM_MODEL, "messages":[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0} # Prepara il pacchetto JSON con temperatura a 0 per la massima precisione.
        res = call_api("chat/completions", payload) # Invia la richiesta al modello tramite la rete protetta.

        if res and 'choices' in res: # Se l'IA ha prodotto una risposta valida...
            answer = re.sub(r'</?end_of_turn>', '', res['choices'][0]['message']['content']).strip() # ...pulisce il testo dai tag tecnici e dagli spazi extra.
            print(f"risposta ai : {answer}") # Mostra la risposta finale all'utente.
        else: # Se la comunicazione con il server è fallita...
            print("risposta ai : Errore di connessione al server") # ...mostra un messaggio di errore pulito.

if __name__ == "__main__": # Controlla se il file viene lanciato come script principale.
    start_pure_chat() # Esegue la funzione della chat.