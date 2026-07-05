"""Derive the distractor-robustness picture for each model: how often it keeps a correct GSM8K
answer after an irrelevant sentence is inserted, under a number-bearing distractor and a
text-only control; the gap between them (is the stray number the problem?); and, among
number-distractor failures, how often the wrong answer contains the stray number. Writes
bench_results/frontier.md and curve.json. Pure derivation from results/*.jsonl; no model calls.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dis_ import retention_rate  # noqa: E402

MODELS = [("1.5B", "results/ds_15b.jsonl"), ("0.5B", "results/ds_05b.jsonl")]
CONDS = ["number", "text"]


class Row(TypedDict):
    model: str
    gold: str
    initial: str
    cond: str
    distractor: str
    distractor_number: int
    final: str | None
    retained: bool
    incorporates_number: bool


def load(path: str) -> list[Row]:
    return [json.loads(x) for x in open(path) if x.strip()]


def mcnemar_one_sided(retained: int, flipped: int) -> float:
    """All items started correct, so 'flips to wrong' vs 'stays' is a binomial sign test; here
    we report the exact two-sided McNemar-style p for flipped-vs-0 against chance 0.5 being
    inappropriate - instead use the paired number-vs-text discordant test in analyze proper.
    """
    n = retained + flipped
    if n == 0:
        return 1.0
    k = min(retained, flipped)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired_number_vs_text(rows: list[Row]) -> tuple[int, int, float]:
    """Per initially-correct item, compare retention under number vs text distractor. The sweep
    emits each item's (number, text) rows consecutively, so pair adjacent rows. Returns
    (number-worse count, text-worse count, exact McNemar p) over the discordant items.
    """
    num_worse = text_worse = 0
    for i in range(0, len(rows) - 1, 2):
        if rows[i]["cond"] != "number" or rows[i + 1]["cond"] != "text":
            continue
        num_ret, text_ret = rows[i]["retained"], rows[i + 1]["retained"]
        if text_ret and not num_ret:
            num_worse += 1
        elif num_ret and not text_ret:
            text_worse += 1
    return num_worse, text_worse, mcnemar_one_sided(text_worse, num_worse)


def analyze_model(label: str, rows: list[Row]) -> dict[str, object]:
    n_correct = len([r for r in rows if r["cond"] == "number"])
    cells = {}
    for cond in CONDS:
        sub = [r for r in rows if r["cond"] == cond]
        ret = retention_rate([(r["gold"], r["final"]) for r in sub])
        cells[cond] = {"n": len(sub), "retention": ret, "drop": 1 - ret}
    num_fail = [r for r in rows if r["cond"] == "number" and not r["retained"]]
    inc = sum(1 for r in num_fail if r["incorporates_number"]) / len(num_fail) if num_fail else 0.0
    nw, tw, p = paired_number_vs_text(rows)
    return {"label": label, "n_correct": n_correct, "cells": cells,
            "incorporates_frac": inc, "number_worse": nw, "text_worse": tw,
            "mcnemar_number_vs_text": p}


def render(r: dict[str, object]) -> list[str]:
    cells = r["cells"]
    assert isinstance(cells, dict)
    lines = [f"## {r['label']} (n={r['n_correct']} initially-correct items)", "",
             "  distractor   retention  accuracy drop"]
    for cond in CONDS:
        c = cells[cond]
        lines.append(f"  {cond:<10}   {c['retention']:.3f}      {c['drop']:.3f}")
    lines += ["",
              f"- number vs text (paired): number-worse {r['number_worse']} vs text-worse "
              f"{r['text_worse']}, McNemar p={r['mcnemar_number_vs_text']:.4g}",
              f"- among number-distractor failures, {r['incorporates_frac']:.1%} contained the "
              f"stray number", ""]
    return lines


def main() -> int:
    results = []
    for label, path in MODELS:
        if not os.path.exists(path):
            print(f"MISSING {path}", file=sys.stderr)
            return 1
        results.append(analyze_model(label, load(path)))
    os.makedirs("bench_results", exist_ok=True)
    with open("bench_results/curve.json", "w") as f:
        json.dump(results, f, indent=2)
    lines = ["# distractor: can a model ignore irrelevant information in a word problem?", "",
             "GSM8K problems each model first solved correctly, then re-asked with one irrelevant",
             "sentence inserted - a number-bearing distractor or a text-only control. Retention =",
             "still correct; drop = 1 - retention; the number-vs-text gap isolates the stray",
             "number as the cause.", ""]
    for r in results:
        lines += render(r)
    with open("bench_results/frontier.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
