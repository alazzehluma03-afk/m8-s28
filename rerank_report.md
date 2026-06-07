# Rerank Report — Module 8 Thursday Stretch

> ~250 words. Replace the placeholder text in each section with your analysis.

## Setup

* Hybrid `k_in`: 50
* Re-ranked `k_out`: 5
* Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
* Hardware (CPU model, RAM, OS): Intel Core i5-1135G7 @ 2.40GHz, 16GB RAM, Windows 11 (Git Bash)

## Metrics Table

| Pipeline | recall@5 | MRR | per-query latency (ms) |
| --- | --- | --- | --- |
| Hybrid (lab baseline) | 0.717 | 0.612 | 14 ms (Stage 1 only) |
| Hybrid + cross-encoder rerank | 0.816 | 0.745 | 159 ms (Stage 1: 14ms + Stage 2: 145ms) |

## When Does Re-Ranking Pay Off?

Re-ranking pays off significantly when semantic nuances and deep contextual alignment matter more than simple keyword matching or dense vector similarity scores. For instance, in complex queries from our 60-pair labeled set where precision at top positions is critical, the hybrid baseline often ranks relevant documents outside the top 5 due to surface-level token mismatches or weak embedding alignment.

The cross-encoder evaluates the query and document jointly rather than independently, capturing complex syntactic dependencies and subtle vocabulary overlap that bi-encoders miss. This architectural change surfaces the gold document to the top positions, yielding a substantial lift in recall@5 (+10 points) and boosting MRR from 0.612 to 0.745.

## Latency Overhead

The cross-encoder introduces a substantial latency overhead of approximately 145 ms per query on our CPU hardware setup. This overhead is entirely independent of total corpus size because the hybrid retriever always filters the search space down to a fixed `k_in = 50` candidates before the cross-encoder runs.

However, the overhead scales strictly and linearly ($O(N)$) with the size of `k_in`. Because the cross-encoder must score each candidate sequentially on the CPU, doubling `k_in` to 100 will predictably double the stage-2 latency to ~290 ms per query, even if the underlying Weaviate database grows to millions of documents.

## At What Corpus Size or Query Volume Does It Stop Being Worth It?

This architecture stops being viable at a query volume exceeding **7 Queries Per Second (QPS)** on a single core, as the 145 ms sequential processing time completely saturates CPU thread capacity ($1 / 0.145 \approx 6.9$).

In terms of corpus size, while expanding the database slows Stage 1 (Hybrid), it does not directly slow Stage 2. However, at a massive corpus scale (e.g., >500,000 documents), a `k_in` of 50 becomes statistically insufficient to capture the gold documents within the initial retrieved pool. To maintain recall, you would be forced to increase `k_in` to 150 or 200, which crashes throughput to less than 2 QPS. At these specific thresholds, transitioning to a GPU-accelerated learned re-ranker or implementing aggressive Redis caching is required.