import streamlit as st
from utils.build_graph import retrieve_from_graph
from utils.advanced_rag import (
    generate_query_variants,
    reciprocal_rank_fusion,
    grade_document,
)
from langchain_core.documents import Document
import requests


def expand_query(query, uri, model):
    """HyDE: append a hypothetical answer to improve recall."""
    try:
        response = requests.post(uri, json={
            "model": model,
            "prompt": f"Generate a concise hypothetical answer (2-3 sentences) to help retrieve relevant documents for: {query}",
            "stream": False
        }, timeout=30).json()
        expansion = response.get("response", "").strip()
        return f"{query}\n{expansion}" if expansion else query
    except Exception:
        return query


def retrieve_documents(query, uri, model, chat_history=""):
    pipeline = st.session_state.retrieval_pipeline
    ensemble = pipeline["ensemble"]

    # ── 1. Base retrieval (RAG-Fusion or single query) ──
    if st.session_state.get("enable_fusion"):
        variants = generate_query_variants(query, uri, model, n=3)
        ranked_lists = []
        for v in variants:
            try:
                ranked_lists.append(ensemble.invoke(v))
            except Exception:
                continue
        docs = reciprocal_rank_fusion(ranked_lists) if ranked_lists else []
    else:
        expanded = expand_query(query, uri, model) if st.session_state.get("enable_hyde") else query
        docs = ensemble.invoke(expanded)

    # ── 2. GraphRAG merge ──
    if st.session_state.get("enable_graph_rag"):
        graph_docs = retrieve_from_graph(query, pipeline["knowledge_graph"], pipeline["doc_chunks"])
        if graph_docs:
            existing = {d.page_content for d in docs}
            for gdoc in graph_docs:
                if gdoc.page_content not in existing:
                    docs.append(gdoc)
                    existing.add(gdoc.page_content)

    # ── 3. Neural reranking ──
    if st.session_state.get("enable_reranking") and pipeline.get("reranker") and docs:
        pairs = [[query, d.page_content] for d in docs]
        scores = pipeline["reranker"].predict(pairs)
        docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]

    max_ctx = st.session_state.get("max_contexts", 3)
    candidates = docs[:max_ctx]

    # ── 4. Corrective RAG (CRAG): grade relevance, drop irrelevant ──
    st.session_state._crag_status = None
    if st.session_state.get("enable_crag") and candidates:
        graded = [(d, grade_document(query, d.page_content, uri, model)) for d in candidates]
        relevant = [d for d, ok in graded if ok]
        if relevant:
            st.session_state._crag_status = ("ok", len(relevant), len(candidates))
            candidates = relevant
        else:
            # Nothing graded relevant — flag low confidence, keep best guess
            st.session_state._crag_status = ("low", 0, len(candidates))
            candidates = candidates[:1]

    return candidates
