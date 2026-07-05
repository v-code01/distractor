# Adversarial review

An attempt to refute each headline claim before publishing. Every objection I could construct
is listed with how the data answers it. Claims that survived are the ones reported.

## Claim A: irrelevant text distracts, small models far more

- **"The inserted sentence changes the math."** The distractor sentences are generic and
  irrelevant (weather, a building's floors, a poster's width) and are inserted after the first
  sentence without altering the problem's quantities or question. The gold answer is unchanged by
  construction, so any drop is the model reacting to irrelevant content.
- **"Maybe the second solve is just noisy."** Decoding is greedy (temperature 0, fixed seed), so
  re-solving the unmodified problem would return the same answer. The drop is caused by the
  inserted sentence, and the text and number conditions are both measured against the same
  initially-correct baseline.

## Claim B: it is not the stray number (the control)

- **"You expected the number to matter and it did not - so the study failed."** The opposite: the
  control is exactly what makes the result informative. The number-vs-text paired test is far from
  significant (p=0.82 and 0.42), which refutes a common intuition and localizes the cause to
  irrelevant content rather than spurious arithmetic. Pre-registering the number hypothesis and
  reporting its falsification is the point.
- **"6% and 24% number-incorporation means the number does get used."** It is reported at its
  measured, minority size. On the 1.5B it is 6% of failures - within noise of coincidence; on the
  0.5B it is 24% but of only ~21 failures. Neither supports the number being the main driver, and
  both are shown rather than rounded away.

## Claim C: distractibility scales with capability

- **"The 0.5B just has a lower baseline, so of course it drops more."** Retention is measured only
  on items the 0.5B itself got right, so its drop is a conditional robustness measure, not a
  reflection of its lower baseline. It keeps a correct answer under an irrelevant sentence only
  about half the time versus the 1.5B's ~85%.
- **"n=39 for the 0.5B is too small."** The 0.5B drop (0.44-0.54) is large and reported with its
  count; the capability claim rests on the three-to-four-fold gap, not a precise value, and the
  1.5B side has n=104.

## Confounds checked

- Distractor sentences are irrelevant and do not alter the problem's quantities; injected at one
  position after the first sentence.
- Greedy decoding, temperature 0, fixed seed: the unmodified problem would re-solve identically.
- Number and text conditions are paired on the same initially-correct items.
- Exact GSM8K last-integer oracle; no judge.

## What this does NOT claim

- Not that numbers never distract - only that, here, a number-bearing distractor is no worse than
  a text-only one, so the number is not the driver.
- Not that larger or instruction-hardened models share this fragility - this is two small models.
- Not a claim about adversarial or on-topic distractors, which could behave differently.
