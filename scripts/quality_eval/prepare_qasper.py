"""Freeze public QASPER questions with answer spans verified in arXiv PDF pages."""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tarfile
import unicodedata
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


logging.getLogger("pypdf").setLevel(logging.ERROR)


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).lower().split())


def answers(qa):
    for row in qa["answers"]:
        item = row["answer"]
        spans = item["extractive_spans"]
        if item["unanswerable"] or len(spans) != 1:
            continue
        answer = spans[0].strip()
        if not 3 <= len(answer) <= 75 or not re.fullmatch(r"[\w .,+%–-]+", answer):
            continue
        for paragraph in item["highlighted_evidence"] + item["evidence"]:
            words = paragraph.split()
            for length in (9, 7, 5):
                for start in range(len(words) - length + 1):
                    anchor = " ".join(words[start:start + length])
                    if re.search(rf"(?<!\w){re.escape(answer)}(?!\w)", anchor, re.I):
                        yield answer, anchor, row["annotation_id"]


def process(item, cache):
    arxiv_id, paper = item
    choices = [(qa, list(answers(qa))) for qa in paper["qas"]]
    if sum(bool(spans) for _, spans in choices) < 2:
        return None
    path = cache / f"arxiv-{arxiv_id}.pdf"
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        if not path.exists():
            pending = path.with_suffix(".partial")
            subprocess.run(["curl.exe", "-fsSL", "--max-time", "30", "--retry", "1",
                            "-o", str(pending), url], check=True, capture_output=True)
            pending.replace(path)
        reader = PdfReader(path)
        if len(reader.pages) > 24:
            return None
        pages = [normalize(page.extract_text() or "") for page in reader.pages]
        if normalize(paper["title"]) not in "".join(pages[:2]):
            return None
        cases = []
        anchors_used = set()
        for qa, options in choices:
            for answer, anchor, annotation in options:
                compact = normalize(anchor)
                if compact in anchors_used:
                    continue
                page_numbers = [n for n, text in enumerate(pages, start=1) if compact in text]
                if len(page_numbers) != 1:
                    continue
                cases.append({"id": f"qasper_{qa['question_id']}", "lang": "en",
                              "query": f"In the paper “{paper['title']}”, {qa['question']}",
                              "original_query": qa["question"], "answer": answer,
                              "evidence": [{"source": path.name, "page": page_numbers[0], "anchor": anchor}],
                              "annotation_id": annotation})
                anchors_used.add(compact)
                break
        if len(cases) >= 2:
            return path, cases, {"arxiv_id": arxiv_id, "url": url, "title": paper["title"], "pages": len(pages)}
    except (OSError, ValueError, PdfReadError, subprocess.SubprocessError) as error:
        print(f"Skipped {arxiv_id}: {error}", flush=True)
    return None


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("baseline_corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with tarfile.open(args.archive) as archive:
        papers = json.load(archive.extractfile("qasper-dev-v0.3.json"))
        training = json.load(archive.extractfile("qasper-train-v0.3.json"))
    cache = args.output / "download-cache"
    corpus = args.output / "corpus"
    cache.mkdir(exist_ok=True, parents=True)
    corpus.mkdir(exist_ok=True, parents=True)
    selected = []
    entries = sorted(papers.items()) + sorted(training.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        for offset in range(0, len(entries), 20):
            for record in pool.map(lambda item: process(item, cache), entries[offset:offset + 20]):
                if record is not None:
                    selected.append(record)
            print(f"Scanned {min(offset + 20, len(entries))} sources, validated {len(selected)}", flush=True)
            if len(selected) >= 90 and any(len(record[1]) > 2 for record in selected[:90]):
                break
    selected = selected[:90]
    if len(selected) != 90:
        raise RuntimeError("Fewer than 90 articles have two usable questions")
    base = json.loads(Path(__file__).with_name("golden-v1.json").read_bytes())
    cases = list(base["cases"])
    names = list(base["corpus_files"])
    for name in names:
        shutil.copyfile(args.baseline_corpus / name, corpus / name)
    manifest = []
    for path, questions, metadata in selected:
        shutil.copyfile(path, corpus / path.name)
        names.append(path.name)
        cases.extend(questions[:2])
        manifest.append({"file": path.name, **metadata, "split": "dev" if metadata["arxiv_id"] in papers else "train",
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    cases.append(next(questions[2] for _, questions, _ in selected if len(questions) > 2))
    assert len(names) == len(set(names)) == 100
    assert len(cases) == len({case["id"] for case in cases}) == 200
    # Score a fixed generation subset covering legacy questions and ten public papers.
    generation_ids = base["generation_ids"] + [record[1][0]["id"] for record in selected[:10]]
    save(args.output / "golden-v2.json", {"version": "rag-pdf-golden-v2", "corpus_files": names,
         "generation_ids": generation_ids, "cases": cases})
    save(args.output / "qasper-source-manifest.json", {"release": "v0.3",
         "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(), "articles": manifest})


if __name__ == "__main__":
    main()
