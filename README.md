# distractor

Drop one irrelevant sentence into a math word problem a model just solved correctly, and does it
still get the answer? This measures how much an off-topic sentence knocks two instruction-tuned
models (Qwen2.5 1.5B and 0.5B) off a GSM8K problem, and - with a control - tests the popular
intuition that it is a stray *number* that lures them. The oracle is exact GSM8K match, no judge.

The headline: **small models are badly distractible, but it is the irrelevant content, not a
stray number, that does it.** Inserting a single off-topic sentence costs the 1.5B about 15% of
the answers it had right and the 0.5B about half of them. Yet a distractor that carries a number
("the store is 7 miles away") is no worse than one that carries none ("the weather was pleasant")
- so the failure is general sensitivity to irrelevant text, and it scales sharply with how weak
the model is.

## Method

- Solve each GSM8K problem; keep, per model, the ones it got right - distraction is only
  meaningful on answers that started correct.
- Re-ask each of those with one irrelevant sentence inserted after the first sentence, under two
  conditions on the same item (paired): a **number** distractor (an irrelevant sentence
  containing a number) and a **text** distractor (an irrelevant sentence with no number, the
  control). The distractor per item is deterministic.
- **Retention** = still correct after the insertion; **drop** = 1 - retention. The
  **number-vs-text** paired comparison isolates the stray number as a cause. A diagnostic checks
  whether a wrong answer contains the stray number.
- Greedy decoding, n=150 GSM8K problems per model (104 initially correct on the 1.5B, 39 on the
  0.5B).

## Pre-registered prediction

Written down before the full run (a pilot showed a 19-point drop on the 1.5B):

> (1) Inserting an irrelevant sentence lowers accuracy - the models cannot fully ignore it.
> (2) A number-bearing distractor hurts more than a text-only one, because the stray number gets
> pulled into the computation. (3) The effect is robust on the 1.5B; the 0.5B is weaker and
> noisier.

Predictions 1 and 3 held. **Prediction 2 was falsified**: the number distractor is not
significantly worse than the text control on either model (McNemar p=0.82 and 0.42). The control
did its job - it refuted the "it's the number" story and showed the damage comes from the
irrelevant sentence itself, reported here as the finding.

## Results

Accuracy drop on initially-correct items when one irrelevant sentence is inserted:

| model | number-distractor drop | text-distractor drop | number vs text (paired) | wrong answers containing the stray number |
|-------|-----------------------:|---------------------:|-------------------------|-------------------------------------------|
| 1.5B (n=104) | 0.154 | 0.135 | 11 vs 9, McNemar p=0.82 | 6.2% |
| 0.5B (n=39)  | 0.538 | 0.436 | 9 vs 5, McNemar p=0.42  | 23.8% |

Full numbers in `bench_results/frontier.md`.

## What this shows

1. **Irrelevant text distracts, and small models far more.** One off-topic sentence costs the
   1.5B about 15% of its correct answers and the 0.5B about half - a three-to-four-fold
   difference. The small model cannot hold onto a solution it just had once the problem statement
   is padded with anything extra.
2. **It is not the stray number.** A number-bearing distractor is no worse than a text-only one
   on either model (the paired number-vs-text test is far from significant). The common intuition
   that models get lured by an irrelevant number is not what drives the drop here; general
   irrelevant content does.
3. **The number is rarely pulled in.** Only 6% of the 1.5B's number-distractor failures literally
   contain the stray number (24% for the 0.5B, on a small count). Even the weaker model mostly
   fails by losing the thread, not by arithmetic-ing the wrong number.
4. **Distractibility is a capability axis.** The same single sentence that barely dents the 1.5B
   halves the 0.5B, so robustness to irrelevant context tracks model strength - a small model's
   headline accuracy overstates how well it holds up on messier, real inputs.

## Limitations and falsifiers

- Two model sizes, one benchmark (GSM8K), a small fixed pool of distractor sentences inserted at
  one position, greedy decoding. Not a claim about larger models or adversarial distractors.
- Retention is measured on initially-correct items by construction; the 0.5B's n is 39, so its
  numbers are directional (reported with the count).
- **Falsifier:** if the number distractor had significantly beaten the text control, the "it is
  not the number" claim would fail. It does not, on either model.
- **Falsifier:** if the small model had not been much more distractible, the capability claim
  would fail. Its drop is three to four times the 1.5B's.
- The adversarial pass that tried to refute each claim is in `REVIEW.md`.

## Reproduce

```bash
./scripts/gate.sh                 # ruff + mypy --strict + pytest + ASCII + independent verify
./reproduce.sh 8081 8082          # rerun both models against two OpenAI-compatible endpoints
```

`tools/verify.py` recomputes the drops and the paired number-vs-text McNemar straight from the
raw JSONL, sharing no code with the analysis, and the ship gate runs it.

## Layout

```
src/dis_.py          GSM8K scoring + number/text distractor pools + injector + retention/diagnostic
tests/test_dis.py    7 unit tests, including injection placement and the number-incorporation check
tools/run_sweep.py   baseline + number/text distractor conditions on initially-correct items
tools/analyze.py     retention and drop by condition, number-vs-text gap, number-incorporation
tools/verify.py      independent recompute of the headline claims (in the gate)
bench_results/       frontier.md + curve.json
claims.toml          every claim tied to its evidence
REVIEW.md            adversarial refutation attempt
```

MIT licensed.
