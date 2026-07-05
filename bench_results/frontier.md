# distractor: can a model ignore irrelevant information in a word problem?

GSM8K problems each model first solved correctly, then re-asked with one irrelevant
sentence inserted - a number-bearing distractor or a text-only control. Retention =
still correct; drop = 1 - retention; the number-vs-text gap isolates the stray
number as the cause.

## 1.5B (n=104 initially-correct items)

  distractor   retention  accuracy drop
  number       0.846      0.154
  text         0.865      0.135

- number vs text (paired): number-worse 11 vs text-worse 9, McNemar p=0.8238
- among number-distractor failures, 6.2% contained the stray number

## 0.5B (n=39 initially-correct items)

  distractor   retention  accuracy drop
  number       0.462      0.538
  text         0.564      0.436

- number vs text (paired): number-worse 9 vs text-worse 5, McNemar p=0.424
- among number-distractor failures, 23.8% contained the stray number

