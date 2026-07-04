"""Measure distractor robustness on GSM8K. First solve each problem; keep the ones the model gets
right. Then re-ask the same problem with one irrelevant sentence inserted, under two conditions -
a number-bearing distractor and a text-only distractor (control) - and record whether the model
still gets it right, and (for the number condition) whether its new answer contains the stray
number. The distractor per item is deterministic. Exact GSM8K oracle, no judge. Writes
results/<model>.jsonl, one row per (initially-correct item, condition).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dis_ import (  # noqa: E402
    correct,
    distractor_for,
    incorporates_number,
    inject,
    parse_gold,
    parse_pred,
)

_SYS = "Solve the math problem step by step. End with 'The answer is N.' where N is a single integer."
_NUM = re.compile(r"\d+")


def solve(url: str, model: str, question: str) -> str | None:
    payload = json.dumps({"model": model,
                          "messages": [{"role": "system", "content": _SYS},
                                       {"role": "user", "content": question}],
                          "temperature": 0.0, "max_tokens": 512, "seed": 1}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            j = json.loads(r.read())
        return parse_pred(str(j["choices"][0]["message"]["content"]))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8081")
    ap.add_argument("--model", default="qwen-ds")
    ap.add_argument("--data", default="data/gsm8k_test.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    items = [json.loads(x) for x in open(args.data)][: args.n]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def baseline(it: dict[str, object]) -> tuple[dict[str, object], str, str | None]:
        gold = parse_gold(str(it["answer"]))
        return it, gold, solve(url, args.model, str(it["question"]))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        base = list(pool.map(baseline, items))
    correct_items = [(it, gold, init) for it, gold, init in base if correct(init, gold)]
    print(f"  {args.model} baseline: {len(correct_items)}/{len(items)} correct", flush=True)

    jobs = []
    for i, (it, gold, init) in enumerate(correct_items):
        for kind in ("number", "text"):
            sent = distractor_for(kind, i, args.seed)
            jobs.append((it, gold, init, kind, sent))

    def run(job: tuple[dict[str, object], str, str | None, str, str]) -> dict[str, object]:
        it, gold, init, kind, sent = job
        final = solve(url, args.model, inject(str(it["question"]), sent))
        m = _NUM.search(sent)
        dnum = int(m.group(0)) if m else -1
        return {"model": args.model, "gold": gold, "initial": init, "cond": kind,
                "distractor": sent, "distractor_number": dnum, "final": final,
                "retained": correct(final, gold),
                "incorporates_number": (kind == "number" and not correct(final, gold)
                                        and incorporates_number(final, dnum))}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, \
            open(args.out, "w", buffering=1) as f:
        for k, row in enumerate(pool.map(run, jobs)):
            f.write(json.dumps(row) + "\n")
            if (k + 1) % 100 == 0:
                print(f"  {args.model} {k + 1}/{len(jobs)}", flush=True)
    print(f"# SWEEP_DONE {args.model}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
