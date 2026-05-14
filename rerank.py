"""Module 8 — Thursday Stretch (Honors Track): Cross-Encoder Re-Ranking.

Add a cross-encoder re-ranking stage to the lab's hybrid retriever and
evaluate the cost/benefit. Cross-encoders score (query, passage) pairs
jointly rather than independently — they produce a more discriminative
ranking, but at a real latency cost.

Use cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
"""

from __future__ import annotations

import weaviate

from retrieval_helpers import hybrid_search

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def cross_encoder_rerank(query: str, candidates: list[dict], k_out: int = 5) -> list[str]:
    """Re-rank a candidate list using a cross-encoder.

    `candidates` is a list of {"doc_id": str, "text": str} (or a similar
    schema providing the text to score). Score each (query, candidate.text)
    pair; sort descending; return the top-`k_out` doc_id strings.

    Hint:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, c["text"]) for c in candidates]
        scores = ce.predict(pairs)
        # argsort descending, take top k_out, map back to doc_id
    """
    # TODO: load CrossEncoder (consider module-level for speed)
    # TODO: build pairs, score with ce.predict, argsort descending, take top k_out
    # TODO: return list of doc_id strings
    raise NotImplementedError("cross_encoder_rerank is not yet implemented")


def rerank_search(
    client: weaviate.Client,
    query: str,
    embedder,
    k_in: int = 50,
    k_out: int = 5,
) -> list[str]:
    """Two-stage retriever: hybrid retrieve k_in, cross-encoder re-rank to k_out.

    Stage 1: hybrid_search(client, query, k_in, embedder, alpha=0.5) -> list[doc_id]
    Stage 2: resolve each doc_id back to its text from Weaviate
    Stage 3: cross_encoder_rerank(query, candidates, k_out)

    Return the ordered list of doc_id strings, length <= k_out.
    """
    # TODO: stage 1: hybrid_search to get k_in candidate doc_ids
    # TODO: resolve each doc_id back to {"doc_id": ..., "text": ...} via Weaviate query
    # TODO: stage 3: cross_encoder_rerank(query, candidates, k_out)
    raise NotImplementedError("rerank_search is not yet implemented")
