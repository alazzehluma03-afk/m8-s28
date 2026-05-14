"""Thursday Stretch (Cross-Encoder Re-Rank) autograder."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import weaviate
from sentence_transformers import SentenceTransformer

from retrieval_helpers import index_corpus_if_needed
from rerank import cross_encoder_rerank, rerank_search

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "retrieval_eval.jsonl")


def _wait(url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if weaviate.Client(url).is_ready():
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Weaviate not ready at {url}")


@pytest.fixture(scope="session", autouse=True)
def setup_corpus():
    _wait(WEAVIATE_URL)
    client = weaviate.Client(WEAVIATE_URL)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index_corpus_if_needed(client, CORPUS_PATH, embedder)


@pytest.fixture(scope="session")
def client():
    return weaviate.Client(WEAVIATE_URL)


@pytest.fixture(scope="session")
def embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def fixture_subset():
    rows = []
    with open(EVAL_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) == 5:
                break
    return rows


def test_cross_encoder_rerank_returns_list_of_str():
    candidates = [
        {"doc_id": "x:1", "text": "git rebase rewrites the commit history"},
        {"doc_id": "x:2", "text": "docker run launches a container"},
        {"doc_id": "x:3", "text": "css display none hides an element"},
    ]
    out = cross_encoder_rerank("how do I rewrite git history", candidates, k_out=2)
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)
    assert len(out) <= 2


def test_rerank_search_returns_list_of_str(client, embedder):
    out = rerank_search(client, "how do I rebase a feature branch", embedder, k_in=20, k_out=5)
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)
    assert len(out) <= 5


def test_rerank_search_recall_floor(client, embedder, fixture_subset):
    hits = 0
    for row in fixture_subset:
        top = rerank_search(client, row["query"], embedder, k_in=50, k_out=10)
        if row["gold_doc_id"] in top[:5]:
            hits += 1
    score = hits / len(fixture_subset)
    assert score >= 0.75, f"rerank recall@5 on fixture = {score}; floor = 0.75"


def test_rerank_report_substance():
    path = os.path.join(os.path.dirname(__file__), "..", "rerank_report.md")
    assert os.path.exists(path)
    body = open(path).read()
    assert len(body) >= 250, f"rerank_report.md too short ({len(body)} chars)"
    lower = body.lower()
    for needle in ("latency", "recall", "cross-encoder"):
        assert needle in lower, f"rerank_report.md must mention {needle!r}"
