from falkordb import FalkorDB
from pyvis.network import Network

# Connessione
db = FalkorDB(host='localhost', port=6379)
graph = db.select_graph('ZucchettiKnowledgeGraph')

def create_interactive_view():
    # Inizializza la rete pyvis
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    
    # Query per prendere tutte le relazioni
    query = "MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name, labels(n)[0], labels(m)[0]"
    res = graph.query(query)
    
    print(f"🎨 Generazione visualizzazione per {len(res.result_set)} relazioni...")

    for row in res.result_set:
        src_name, rel_type, dst_name, src_label, dst_label = row
        
        # Colore basato sul tipo di nodo
        color_map = {"Concept": "#ed1c24", "Detail": "#00aef0"}
        
        # Aggiungi nodi
        net.add_node(src_name, label=src_name, color=color_map.get(src_label, "grey"))
        net.add_node(dst_name, label=dst_name, color=color_map.get(dst_label, "grey"))
        
        # Aggiungi freccia
        net.add_edge(src_name, dst_name, title=rel_type, label=rel_type)

    # Imposta la fisica per un movimento fluido
    net.toggle_physics(True)
    
    # Salva e apri
    output_file = "zucchetti_graph.html"
    net.save_graph(output_file)
    print(f"✨ Visualizzazione salvata con successo in: {output_file}")
    print("👉 Apri questo file con il tuo browser (Chrome/Edge) per vedere il grafo!")

if __name__ == "__main__":
    create_interactive_view()