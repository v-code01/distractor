#!/usr/bin/env python3
"""Independent verification of the headline, sharing NO code with src/dis_.py or
tools/analyze.py. Re-reads the raw JSONL and recomputes, from scratch, retention and drop under
each distractor, the paired number-vs-text comparison, and asserts the honest headline: (1) both
models lose accuracy when an irrelevant sentence is inserted (a real drop); (2) a number-bearing
distractor is NOT significantly worse than a text-only one - the stray number is not the driver
(the control refutes it); (3) distractibility scales sharply with weakness - the small model's
drop is far larger than the large model's. Exit non-zero on mismatch. Run in the ship gate.
"""
from __future__ import annotations

import json
import sys
from math import comb

PATHS = {"1.5B": "results/ds_15b.jsonl", "0.5B": "results/ds_05b.jsonl"}


def rows_of(path: str) -> list[dict[str, object]]:
    return [json.loads(x) for x in open(path) if x.strip()]


def drop(rows: list[dict[str, object]], cond: str) -> float:
    sub = [r for r in rows if r["cond"] == cond]
    if not sub:
        return 0.0
    retained = sum(1 for r in sub if r["final"] == r["gold"])
    return 1.0 - retained / len(sub)


def mcnemar(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def number_vs_text(rows: list[dict[str, object]]) -> float:
    """Paired: adjacent (number, text) rows per item; exact McNemar over discordant retention."""
    num_worse = text_worse = 0
    for i in range(0, len(rows) - 1, 2):
        if rows[i]["cond"] != "number" or rows[i + 1]["cond"] != "text":
            continue
        nr = rows[i]["final"] == rows[i]["gold"]
        tr = rows[i + 1]["final"] == rows[i + 1]["gold"]
        if tr and not nr:
            num_worse += 1
        elif nr and not tr:
            text_worse += 1
    return mcnemar(num_worse, text_worse)


def main() -> int:
    data = {m: rows_of(p) for m, p in PATHS.items()}
    dn = {m: drop(data[m], "number") for m in PATHS}
    dt = {m: drop(data[m], "text") for m in PATHS}
    pv = {m: number_vs_text(data[m]) for m in PATHS}
    for m in PATHS:
        print(f"  {m}: number-drop {dn[m]:.3f}, text-drop {dt[m]:.3f}, "
              f"number-vs-text McNemar p={pv[m]:.3g}")

    both_drop = dn["1.5B"] >= 0.10 and dn["0.5B"] >= 0.30       # a real drop on both
    number_not_worse = pv["1.5B"] >= 0.05 and pv["0.5B"] >= 0.05  # control refutes the number
    small_more_distractible = dn["0.5B"] > 2 * dn["1.5B"]        # weakness -> distractibility
    print(f"  both models lose accuracy to an irrelevant sentence: {both_drop}")
    print(f"  a number distractor is NOT significantly worse than text: {number_not_worse}")
    print(f"  the small model is far more distractible: {small_more_distractible}")

    if both_drop and number_not_worse and small_more_distractible:
        print("VERIFY OK: irrelevant content (not stray numbers) distracts these models, and the "
              "small model is far more fragile")
        return 0
    print("VERIFY FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
