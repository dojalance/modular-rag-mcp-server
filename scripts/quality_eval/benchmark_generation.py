"""Generate from fixed retrieved passages and score with a separate local Ragas judge."""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall
from ragas.run_config import RunConfig


MODES = ("rrf", "rrf_cross_encoder")
GENERATOR = "qwen3-1.7b"
JUDGE = "rag-judge-qwen3-4b"
GENERATOR_URL = "http://127.0.0.1:18081/v1"
JUDGE_URL = "http://127.0.0.1:18083/v1"
ABSTAIN = {"zh": "资料未提供答案", "en": "The supplied documents do not answer this question"}
PROMPTS = {
    "zh": "只用给出的段落，直接用中文回答。对事实在句末注明段落编号如 [1]。若段落没有所需信息，统一回答「资料未提供答案」。",
    "en": "Answer briefly in English using only the provided passages and cite passage numbers such as [1]. If they do not contain the answer, reply exactly: The supplied documents do not answer this question.",
}


def save(file, value):
    staging = file.with_suffix(".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(file)


class JudgeCapture(BaseCallbackHandler):
    def __init__(self):
        self.calls = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.calls[str(run_id)] = {"prompt": [[message.model_dump() for message in batch] for batch in messages]}

    def on_llm_end(self, response, *, run_id, **kwargs):
        self.calls.setdefault(str(run_id), {}).update({"output": [
            [generation.message.model_dump() for generation in group] for group in response.generations
        ]})

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.calls.setdefault(str(run_id), {}).update({"error": str(error)})


def generate(fixture, rows, artifacts, ids, existing):
    cases = {case["id"]: case for case in fixture["cases"]}
    ranking = {(item["id"], item["mode"]): item["top5"] for item in rows if item["repetition"] == 1}
    results = list(existing)
    done = {(row["mode"], row["id"]) for row in results}
    with httpx.Client(timeout=60, trust_env=False) as client:
        for mode in MODES:
            for case_id in ids:
                if (mode, case_id) in done:
                    continue
                case = cases[case_id]
                contexts = [item["text"] for item in ranking[case_id, mode]]
                passages = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(contexts, start=1))
                messages = [{"role": "system", "content": PROMPTS[case["lang"]]},
                            {"role": "user", "content": f"{case['query']}\n\n{passages}"}]
                start = time.perf_counter()
                response = client.post(GENERATOR_URL + "/chat/completions", json={
                    "model": GENERATOR, "messages": messages, "temperature": 0, "max_tokens": 192,
                }).raise_for_status().json()
                elapsed = (time.perf_counter() - start) * 1000
                answer = response["choices"][0]["message"]["content"].strip()
                results.append({"id": case_id, "mode": mode, "query": case["query"], "answer": answer,
                                "reference": case["answer"], "contexts": contexts, "answerable": bool(case["evidence"]),
                                "abstained": answer.startswith(ABSTAIN[case["lang"]]),
                                "prompt": messages, "usage": response.get("usage"), "elapsed_ms": elapsed})
                save(artifacts / "generation-rows.json", results)
                print(f"Generated {mode} {case_id}", flush=True)
    return results


def run_judge(results, artifacts):
    scored = [row for row in results if row["answerable"]]
    dataset = EvaluationDataset(samples=[SingleTurnSample(user_input=row["query"], response=row["answer"],
                                   reference=row["reference"], retrieved_contexts=row["contexts"]) for row in scored])
    capture = JudgeCapture()
    judge = ChatOpenAI(model=JUDGE, base_url=JUDGE_URL, api_key="local-no-key", temperature=0,
                       max_tokens=768, max_retries=0, timeout=75, callbacks=[capture])
    run_config = RunConfig(timeout=85, max_retries=2, max_workers=1, seed=42)
    try:
        evaluation = evaluate(dataset, metrics=[Faithfulness(), LLMContextRecall()], llm=LangchainLLMWrapper(judge),
                              raise_exceptions=False, run_config=run_config, show_progress=False)
        for row, values in zip(scored, evaluation.scores, strict=True):
            row["ragas"] = {key: float(value) if math.isfinite(value) else None for key, value in values.items()}
    finally:
        save(artifacts / "judge-calls.json", capture.calls)
        save(artifacts / "generation-rows.json", results)
    metrics = {}
    for mode in MODES:
        rows = [row for row in scored if row["mode"] == mode]
        control = [row for row in results if row["mode"] == mode and not row["answerable"]]
        metrics[mode] = {"answerable_count": len(rows), "unanswerable_count": len(control),
                         "abstained_on_unanswerable": sum(row["abstained"] for row in control)}
        for key in ("faithfulness", "context_recall"):
            values = [row["ragas"].get(key) for row in rows if row["ragas"].get(key) is not None]
            metrics[mode][key] = {"mean": statistics.mean(values) if values else None, "valid": len(values)}
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("retrieval", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("existing", type=Path, nargs="?")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--judge-frozen-subset", action="store_true")
    args = parser.parse_args()
    retrieval, artifacts = args.retrieval.resolve(), args.artifacts.resolve()
    retrieval_summary = json.loads((retrieval / "retrieval-summary.json").read_bytes())
    version = retrieval_summary["version"]
    raw_fixture = (retrieval / f"golden-{version.rsplit('-', 1)[-1]}.json").read_bytes()
    fixture_sha256 = hashlib.sha256(raw_fixture).hexdigest()
    if fixture_sha256 != retrieval_summary["fixture_sha256"]:
        raise ValueError("Frozen fixture differs from the retrieval benchmark")
    fixture = json.loads(raw_fixture)
    ids = [case["id"] for case in fixture["cases"]] if args.all_questions else fixture["generation_ids"]
    if args.existing and not args.generate_only:
        generated = json.loads(args.existing.read_bytes())
    else:
        rows = json.loads((retrieval / "retrieval-rows.json").read_bytes())
        existing = json.loads(args.existing.read_bytes()) if args.existing else []
        generated = generate(fixture, rows, artifacts, ids, existing)
    expected = {(mode, case) for mode in MODES for case in ids}
    if {(row["mode"], row["id"]) for row in generated} != expected or len(generated) != len(expected):
        raise ValueError("Generation rows do not match frozen case scope")
    if args.generate_only:
        return 0
    if args.judge_frozen_subset:
        ids = fixture["generation_ids"]
        generated = [row for row in generated if row["id"] in ids]
    summary = {"fixture_sha256": fixture_sha256, "generation_ids": ids,
               "generator": GENERATOR, "judge": JUDGE, "metrics": run_judge(generated, artifacts)}
    save(artifacts / "generation-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if all(group[name]["valid"] == group["answerable_count"] for group in summary["metrics"].values()
                    for name in ("faithfulness", "context_recall")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
