# Rerank Report — Module 8 Thursday Stretch

> ~250 words. Replace the placeholder text in each section with your analysis.

## Setup

- Hybrid `k_in`: _50_
- Re-ranked `k_out`: _5_
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Hardware (CPU model, RAM, OS): _your environment_

## Metrics Table

| Pipeline | recall@5 | MRR | per-query latency (ms) |
|---|---|---|---|
| Hybrid (lab baseline) | _your number_ | _your number_ | _stage 1 only_ |
| Hybrid + cross-encoder rerank | _your number_ | _your number_ | _stage 1 + stage 2_ |

Report stage-1 (hybrid retrieve) and stage-2 (cross-encoder score 50 pairs)
latency separately.

## When Does Re-Ranking Pay Off?

Cite specific queries from the labeled set where re-ranking surfaces the
gold doc when hybrid did not (or vice versa).

## Latency Overhead

How much does the cross-encoder add per query? Is the overhead consistent or
does it scale with `k_in`? What about with corpus size (the cross-encoder
runs on `k_in` pairs regardless of corpus size, but the hybrid retrieve
slows with larger corpora)?

## At What Corpus Size or Query Volume Does It Stop Being Worth It?

Estimate the cross-over: at what QPS does the cross-encoder become the
bottleneck? At what corpus size do you need a different approach (learned
re-ranker, aggressive caching)? Use specific numbers — "it depends" is not
an answer.
