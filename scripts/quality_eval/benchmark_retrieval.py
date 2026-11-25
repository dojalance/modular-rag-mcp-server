"""Run a frozen PDF retrieval ablation against local GPU models and Chroma."""

import argparse
import hashlib
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import yaml
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
sys.path.insert(0, str(SOURCE))

from scripts.query import _build_components
from src.core import settings as settings_module
from src.core.query_engine.reranker import CoreReranker
from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Document
from src.ingestion.pipeline import IngestionPipeline
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


COLLECTION = "frozen_pdf_quality_v1"
MODES = ("bm25", "dense", "rrf", "rrf_cross_encoder")
REPEATS = 2
TOP_K = 5
CANDIDATES = 10


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save(path, data):
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def norm(value):
    return "".join(unicodedata.normalize("NFKC", value).lower().split())


def has_evidence(text, label):
    text = norm(text)
    return all(norm(label[key]) in text for key in ("anchor", "also") if key in label)


class PageTextExtraction:
    """Use each PDF text-layer page as the unit passed to the existing splitter."""

    def __init__(self, loader, chunker, pages):
        self.loader = loader
        self.chunker = chunker
        self.pages = pages

    def load(self, file_path):
        path = Path(file_path)
        if path not in self.pages:
            return self.loader.load(file_path)
        digest = sha(path)
        return Document(id=f"doc_{digest[:16]}", text="\n\n".join(self.pages[path]),
                        metadata={"source_path": str(path), "doc_type": "pdf", "doc_hash": digest,
                                  "text_extraction_method": "pypdf_page_text"})

    def split_document(self, document):
        path = Path(document.metadata["source_path"])
        if path not in self.pages:
            return self.chunker.split_document(document)
        chunks = []
        for number, text in enumerate(self.pages[path], start=1):
            if not text.strip():
                continue
            page = replace(document, id=f"{document.id}_page_{number}", text=text,
                           metadata={**document.metadata, "page": number})
            for chunk in self.chunker.split_document(page):
                chunk.source_ref = document.id
                chunk.metadata.update(source_ref=document.id, chunk_index=len(chunks))
                chunks.append(chunk)
        return chunks


class CrossEncoderProcess:
    def __init__(self, process):
        self.process = process
        self.ready = json.loads(process.stdout.readline())

    def predict(self, pairs):
        self.process.stdin.write(json.dumps(pairs, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        scores = json.loads(self.process.stdout.readline())
        if len(scores) != len(pairs) or not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Cross-Encoder returned invalid scores")
        return scores


@contextmanager
def cross_encoder(cuda_python, weights, artifacts):
    with (artifacts / "cross-encoder.stderr.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen([str(cuda_python), str(HERE / "cross_worker.py"), str(weights)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                              text=True, encoding="utf-8", env=dict(os.environ, PYTHONUTF8="1")) as process:
            try:
                yield CrossEncoderProcess(process)
            finally:
                process.stdin.close()


def configure(template, artifacts, chroma_dir, ingestion_batch_size):
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["vector_store"]["persist_directory"] = str(chroma_dir)
    config["ingestion"].update(chunk_size=450, chunk_overlap=100, batch_size=ingestion_batch_size)
    config["retrieval"].update(dense_top_k=CANDIDATES, sparse_top_k=CANDIDATES, fusion_top_k=CANDIDATES)
    config["rerank"].update(enabled=True, provider="cross_encoder", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    path = artifacts / "settings.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_settings(path)


def ingest(frozen, corpus, settings, artifacts, collection, resume, page_text_files):
    files = {name: corpus / name for name in frozen["corpus_files"]}
    pages = {name: [(page.extract_text() or "") for page in PdfReader(path).pages] for name, path in files.items()}
    for case in frozen["cases"]:
        for label in case["evidence"]:
            assert has_evidence(pages[label["source"]][label["page"] - 1], label), (case["id"], label)
    journal = artifacts / "corpus-ingestion.json"
    results = json.loads(journal.read_bytes()) if resume else []
    if resume:
        prefix = list(files)[:len(results)]
        if [row["file"] for row in results] != prefix:
            raise ValueError("Existing ingestion journal does not match the frozen corpus order")
        for row in results:
            if (not row["result"]["success"] or row["result"]["chunk_count"] <= 0
                    or row["sha256"] != sha(files[row["file"]])):
                raise ValueError(f"Existing ingestion row failed verification: {row['file']}")
        print(f"Resuming after {len(results)} verified documents", flush=True)
    extractor = IngestionPipeline(settings, collection=collection, force=resume)
    if page_text_files:
        extraction = PageTextExtraction(extractor.loader, extractor.chunker,
                                        {files[name]: pages[name] for name in page_text_files})
        extractor.loader = extractor.chunker = extraction
    for name in list(files)[len(results):]:
        path = files[name]
        trace = TraceContext(trace_type="ingestion")
        result = extractor.run(str(path), trace=trace)
        results.append({"file": name, "sha256": sha(path), "pages": len(pages[name]),
                        "extraction": "pypdf_page_text" if name in page_text_files else "markitdown",
                        "result": result.to_dict(), "trace": trace.to_dict()})
        save(journal, results)
        print(f"Ingest {name}: success={result.success} chunks={result.chunk_count}", flush=True)
    if not all(entry["result"]["success"] and entry["result"]["chunk_count"] > 0 for entry in results):
        raise RuntimeError("At least one frozen PDF failed ingestion or returned no chunks")
    return results


def indexed_evidence(frozen, settings, artifacts, collection):
    store = VectorStoreFactory.create(settings, collection_name=collection)
    rows = store.collection.get(include=["documents", "metadatas"])
    chunks = [{"id": key, "source": Path(meta["source_path"]).name, "text": text,
               "page": meta.get("page") if meta.get("text_extraction_method") == "pypdf_page_text" else None}
              for key, meta, text in zip(rows["ids"], rows["metadatas"], rows["documents"], strict=True)]
    save(artifacts / "indexed-chunks.json", chunks)
    truth = {}
    for case in frozen["cases"]:
        truth[case["id"]] = [
            {**label, "chunk_ids": [row["id"] for row in chunks if row["source"] == label["source"]
                                    and row["page"] in (None, label["page"])
                                    and has_evidence(row["text"], label)]}
            for label in case["evidence"]
        ]
    save(artifacts / "resolved-evidence.json", truth)
    return chunks, truth


def evaluate(frozen, truth, hybrid, reranker, artifacts):
    evaluator = CustomEvaluator()
    base_config = hybrid.config

    def one(case, mode, repetition):
        hybrid.config = replace(base_config, enable_dense=mode != "bm25", enable_sparse=mode != "dense")
        query = case["query"]
        trace = TraceContext(trace_type="query")
        trace.metadata.update(case=case["id"], mode=mode, repetition=repetition)
        started = time.perf_counter()
        detail = hybrid.search(query, top_k=CANDIDATES, trace=trace, return_details=True)
        if detail.dense_error or detail.sparse_error:
            raise RuntimeError((mode, detail.dense_error, detail.sparse_error))
        ranking = detail.results
        fallback = False
        if mode == "rrf_cross_encoder":
            reranked = reranker.rerank(query, ranking, top_k=TOP_K, trace=trace)
            ranking, fallback = reranked.results, reranked.used_fallback
        ranking = ranking[:TOP_K]
        duration_ms = (time.perf_counter() - started) * 1000
        ids = [entry.chunk_id for entry in ranking]
        evidence = truth[case["id"]]
        relevant = sorted({key for label in evidence for key in label["chunk_ids"]})
        scores = None
        if evidence:
            scores = evaluator.evaluate(query, ids, ground_truth=relevant) if ids else {"hit_rate": 0.0, "mrr": 0.0}
            scores["evidence_recall"] = sum(bool(set(ids) & set(label["chunk_ids"])) for label in evidence) / len(evidence)
            scores["hit_at_1"] = int(bool(ids) and ids[0] in relevant)
        return {"id": case["id"], "mode": mode, "repetition": repetition, "elapsed_ms": duration_ms,
                "scores": scores, "fallback": fallback,
                "top5": [{"id": entry.chunk_id, "source": Path(entry.metadata["source_path"]).name,
                          "score": entry.score, "text": entry.text} for entry in ranking],
                "trace": trace.to_dict()}

    warmup = {"id": "warmup", "query": "information retrieval overview"}
    truth["warmup"] = []
    for mode in MODES:
        one(warmup, mode, 0)
    rows = []
    for repeat in range(1, REPEATS + 1):
        for index, case in enumerate(frozen["cases"]):
            offset = (index + repeat - 1) % len(MODES)
            for mode in (*MODES[offset:], *MODES[:offset]):
                rows.append(one(case, mode, repeat))
            print(f"Repeat {repeat}: {case['id']}", flush=True)
        save(artifacts / "retrieval-rows.json", rows)
    result = {}
    for mode in MODES:
        subset = [row for row in rows if row["mode"] == mode]
        durations = sorted(row["elapsed_ms"] for row in subset)
        scores = [row["scores"] for row in subset if row["scores"] is not None]
        result[mode] = {"quality_observations": len(scores), "latency_observations": len(durations),
                        "p50_ms": statistics.median(durations), "p95_ms": durations[math.ceil(.95 * len(durations)) - 1],
                        "fallbacks": sum(row["fallback"] for row in subset),
                        **{key: statistics.mean(score[key] for score in scores) for key in scores[0]}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--fixture", type=Path, default=HERE / "golden-v1.json")
    parser.add_argument("--corpus", type=Path, default=SOURCE.parent / "corpus")
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--resume-ingestion", action="store_true")
    parser.add_argument("--chroma-dir", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--ingestion-batch-size", type=int, default=8)
    parser.add_argument("--page-text-manifest", type=Path)
    args = parser.parse_args()
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    workspace = SOURCE.parents[2]
    lab = SOURCE.parent
    artifacts = args.artifacts.resolve()
    fixture = args.fixture.resolve()
    corpus = args.corpus.resolve()
    state_dir = (args.state_dir or artifacts).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = (args.chroma_dir or artifacts / "chroma").absolute()
    if args.resume_ingestion and sha(artifacts / fixture.name) != sha(fixture):
        raise ValueError("Cannot resume with a different frozen fixture")
    frozen = json.loads(fixture.read_text(encoding="utf-8"))
    page_text_files = set()
    if args.page_text_manifest:
        for article in json.loads(args.page_text_manifest.read_bytes())["articles"]:
            name = article["file"]
            if name not in frozen["corpus_files"]:
                continue
            if sha(corpus / name) != article["sha256"]:
                raise ValueError(f"Unexpected PDF hash: {name}")
            page_text_files.add(name)
    (artifacts / fixture.name).write_bytes(fixture.read_bytes())
    config = lab.parent / "phase05-slot-integration/rag-mcp-corpus-v1/server/config/settings.yaml"
    settings = configure(config, artifacts, chroma_dir, args.ingestion_batch_size)
    settings_module.REPO_ROOT = state_dir
    os.chdir(state_dir)
    ingested = ingest(frozen, corpus, settings, artifacts, args.collection, args.resume_ingestion, page_text_files)
    chunks, truth = indexed_evidence(frozen, settings, artifacts, args.collection)
    if len(chunks) != sum(entry["result"]["chunk_count"] for entry in ingested):
        raise RuntimeError("The vector index does not match the ingestion journal")
    weights = workspace / ".runtime/rag-cross-encoder"
    cuda_python = workspace / ".runtime/phase04-env/Scripts/python.exe"
    hybrid, _ = _build_components(replace(settings, rerank=replace(settings.rerank, enabled=False)), args.collection)
    with cross_encoder(cuda_python, weights, artifacts) as worker:
        ranker = CoreReranker(settings, reranker=CrossEncoderReranker(settings, model=worker))
        scores = evaluate(frozen, truth, hybrid, ranker, artifacts)
    summary = {"version": frozen["version"], "fixture_sha256": sha(fixture), "corpus_files": len(ingested),
               "indexed_chunks": len(chunks), "queries": len(frozen["cases"]), "repeats": REPEATS,
               "collection": args.collection, "pages": sum(entry["pages"] for entry in ingested),
               "page_text_files": len(page_text_files),
               "unmapped_evidence": [case for case, labels in truth.items() if any(not label["chunk_ids"] for label in labels)],
               "cross_encoder": {**worker.ready, **json.loads((weights / "revision.json").read_text(encoding="utf-8")),
                                 "weights_sha256": sha(weights / "model.safetensors")}, "results": scores,
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()}
    save(artifacts / "retrieval-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
