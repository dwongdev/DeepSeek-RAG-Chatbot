import networkx as nx
import re


def build_knowledge_graph(docs):
    G = nx.Graph()
    for doc in docs:
        entities = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', doc.page_content)
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                G.add_edge(entities[i], entities[i + 1], source=doc.page_content[:200])
    return G


def retrieve_from_graph(query, G, doc_chunks, top_k=3):
    """Return actual document text chunks related to graph nodes matching the query."""
    query_words = query.lower().split()
    matched_nodes = [node for node in G.nodes if any(word in node.lower() for word in query_words)]

    if not matched_nodes:
        return []

    related_nodes = set()
    for node in matched_nodes:
        related_nodes.update(G.neighbors(node))

    # Find doc chunks that contain any of the matched/related entity names
    all_entities = set(matched_nodes) | related_nodes
    results = []
    seen = set()
    for chunk in doc_chunks:
        text = chunk.page_content
        if any(entity.lower() in text.lower() for entity in all_entities):
            if text not in seen:
                seen.add(text)
                results.append(chunk)
        if len(results) >= top_k:
            break

    return results
