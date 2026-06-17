"""
Advanced RAG techniques: Contextual Retrieval, RAG-Fusion (RRF),
Corrective RAG (CRAG) grading, and Semantic Caching helpers.

All functions are pure / stateless and degrade gracefully (return safe
fallbacks) if the LLM call fails, so the main pipeline never crashes.
"""
import requests
import math


def _ollama_generate(uri, model, prompt, temperature=0.0, timeout=60):
    """Single non-streamed Ollama completion. Returns '' on failure."""
    try:
        r = requests.post(
            uri,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        return r.json().get("response", "").strip()
    except Exception:
        return ""


# ─── 1. Contextual Retrieval (Anthropic) ──────────────────────────────────────
def contextualize_chunk(chunk_text, doc_summary, uri, model):
    """Generate a short situating context for a chunk given its document summary."""
    prompt = (
        "You are helping index a document for search.\n"
        f"<document_summary>\n{doc_summary}\n</document_summary>\n"
        f"<chunk>\n{chunk_text}\n</chunk>\n\n"
        "Write a SINGLE short sentence (max 25 words) that situates this chunk "
        "within the document so it can be retrieved on its own. "
        "Output only that sentence, nothing else."
    )
    ctx = _ollama_generate(uri, model, prompt, temperature=0.0, timeout=60)
    # Keep it tidy and bounded
    ctx = ctx.replace("\n", " ").strip()
    return ctx[:300]


# ─── 2. RAG-Fusion: multi-query + Reciprocal Rank Fusion ──────────────────────
def generate_query_variants(query, uri, model, n=3):
    """Return [original] + up to n reworded search queries."""
    prompt = (
        f"Generate {n} alternative search queries that capture different phrasings "
        f"or sub-aspects of the question below. One query per line, no numbering.\n\n"
        f"Question: {query}\n\nQueries:"
    )
    out = _ollama_generate(uri, model, prompt, temperature=0.4, timeout=45)
    variants = []
    for line in out.splitlines():
        v = line.strip().lstrip("0123456789.-*) ").strip()
        if v and v.lower() != query.lower():
            variants.append(v)
    return [query] + variants[:n]


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Merge multiple ranked Document lists into one via RRF."""
    scores = {}
    doc_map = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ordered]


# ─── 3. Corrective RAG (CRAG): relevance grading ──────────────────────────────
def grade_document(query, doc_text, uri, model):
    """Return True if the document is relevant to the query (LLM judge)."""
    prompt = (
        "You are a strict relevance grader. Decide whether the document contains "
        "information useful to answer the question.\n"
        f"Question: {query}\n"
        f"Document: {doc_text[:1200]}\n\n"
        "Answer with a single word: 'yes' or 'no'."
    )
    ans = _ollama_generate(uri, model, prompt, temperature=0.0, timeout=30).lower()
    return ans.startswith("y") or "yes" in ans[:6]


# ─── 4. Semantic Cache helpers ────────────────────────────────────────────────
def cosine_similarity(a, b):
    """Cosine similarity between two equal-length float lists."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
