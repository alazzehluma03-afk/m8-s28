"""Module 8 — Thursday Stretch (Honors Track): Cross-Encoder Re-Ranking.

Add a cross-encoder re-ranking stage to the lab's hybrid retriever and
evaluate the cost/benefit. Cross-encoders score (query, passage) pairs
jointly rather than independently — they produce a more discriminative
ranking, but at a real latency cost.

Use cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
"""

from __future__ import annotations

import weaviate
from sentence_transformers import CrossEncoder
from retrieval_helpers import hybrid_search

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

ce_model = CrossEncoder(CROSS_ENCODER_MODEL)


def cross_encoder_rerank(query: str, candidates: list[dict], k_out: int = 5) -> list[str]:
    """Re-rank a candidate list using a cross-encoder.

    `candidates` is a list of {"doc_id": str, "text": str} (or a similar
    schema providing the text to score). Score each (query, candidate.text)
    pair; sort descending; return the top-`k_out` doc_id strings.
    """
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    
    scores = ce_model.predict(pairs)
    
    scored_candidates = []
    for idx, score in enumerate(scores):
        scored_candidates.append({
            "doc_id": candidates[idx]["doc_id"],
            "score": float(score)
        })
    
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = scored_candidates[:k_out]
    
    return [c["doc_id"] for c in top_candidates]


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
    candidate_ids = hybrid_search(client, query, k_in, embedder, alpha=0.5)
    
    if not candidate_ids:
        return []

    class_name = "Document"
    try:
        schema = client.schema.get()
        classes = [c["class"] for c in schema.get("classes", [])]
        if classes:
            if "Paragraph" in classes:
                class_name = "Paragraph"
            elif "Document" in classes:
                class_name = "Document"
            else:
                class_name = classes[0]
    except Exception:
        pass

    candidates = []
    
    for doc_id in candidate_ids:
        try:
            result = (
                client.query.get(class_name, ["doc_id", "text"])
                .with_where({
                    "path": ["doc_id"],
                    "operator": "Equal",
                    "valueString": doc_id
                })
                .do()
            )
            
            data = result["data"]["Get"][class_name]
            if data:
                candidates.append({
                    "doc_id": doc_id,
                    "text": data[0]["text"]
                })
                continue
        except Exception:
            pass

        try:
            res = client.data_object.get_by_id(doc_id, class_name=class_name)
            if res:
                props = res.get("properties", {})
                candidates.append({
                    "doc_id": props.get("doc_id", doc_id),
                    "text": props.get("text", "")
                })
        except Exception:
            continue

    final_rankings = cross_encoder_rerank(query, candidates, k_out)
    
    return final_rankings