# Module 8 Stretch (Thu) — Cross-Encoder Re-Ranking (Honors Track)

Add a cross-encoder re-ranking stage to the lab's hybrid retriever and
evaluate the cost/benefit.

Full assignment instructions are on the **Thursday Stretch page** in
TalentLMS → Module 8 → Thursday Stretch.

## Setup

1. Bring up Weaviate and ingest the corpus (same pattern as the lab + Tue
   stretch):
   ```bash
   docker run -d --name weaviate-stretch-thu \
     -p 8080:8080 \
     -e DEFAULT_VECTORIZER_MODULE=none \
     -e ENABLE_MODULES= \
     -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
     semitechnologies/weaviate:1.24.10
   pip install -r requirements.txt
   python ingest.py
   ```

2. Implement `rerank.py` (`cross_encoder_rerank` + `rerank_search`).

3. Run the autograder locally:
   ```bash
   pytest tests/ -v
   ```

4. Fill in `rerank_report.md` (~250 words).

5. Branch `stretch-cross-encoder-rerank`, commit, open PR, paste PR URL into
   TalentLMS.

## Files

- `rerank.py` — your implementation
- `retrieval_helpers.py` — provided: `bm25_search`, `dense_search`, `hybrid_search`,
  `index_corpus_if_needed`
- `ingest.py` — runnable driver
- `rerank_report.md` — your cost/benefit writeup
- `data/corpus.jsonl` — same technical-Q&A corpus as the lab
- `data/retrieval_eval.jsonl` — same 60-pair labeled set as the lab
- `tests/test_rerank.py` — autograder
- `LICENSE` + `ATTRIBUTION.md` — corpus license and attribution

## Resubmissions

Accepted through Saturday of the assignment week.

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms. Corpus content is governed separately by the CC BY-SA notices in [ATTRIBUTION.md](ATTRIBUTION.md) and `data/LICENSE`.
