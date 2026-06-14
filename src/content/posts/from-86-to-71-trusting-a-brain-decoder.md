---
title: "From 86% to 71%: learning to trust a brain decoder's accuracy"
description: "I trained a tiny CNN to read MEG brain signals on Colab. The interesting part wasn't the model — it was how stratified splits, checkpoint hygiene, and cross-validation deflated a flattering 86% into an honest 71% ± 5%, and revealed exactly where the model wins."
pubDatetime: 2026-06-14T12:30:00+05:30
draft: false
tags:
  - machine-learning
  - neurotech
  - evaluation
---

I set out to train a small neural network to **read someone's brain** — given half a second of MEG signal, guess whether they just heard a sound on the left/right or saw something on the left/right. The model worked. But the part worth writing down isn't the model; it's how the *number I reported* swung from **65.5% → 86.2% → 71% ± 5%** as I fixed one evaluation mistake after another. The honest answer was never 86%.

This ran on Google Colab (see [how I use Colab](/posts/how-i-use-google-colab) for the setup), using Meta's [neuroai](https://github.com/facebookresearch/neuroai) stack — `neuralset` for data, `neuraltrain` + PyTorch Lightning for training, and an [EEGNet](https://doi.org/10.1088/1741-2552/aace8c) from `braindecode`.

## 🧠 The task

The data is the [MNE sample dataset](https://mne.tools/stable/documentation/datasets.html#sample): one person in an MEG scanner (≈300 magnetic-field sensors around the head), shown a stream of stimuli about every 0.75 s. Four stimulus types, interleaved:

| Code | Stimulus | Axis |
|------|----------|------|
| 0 | Auditory — **left** ear | sound · side |
| 1 | Auditory — **right** ear | sound · side |
| 2 | Visual — **left** field | sight · side |
| 3 | Visual — **right** field | sight · side |

Each training example is a slice of brain activity around one stimulus — `(306 channels × 60 timepoints)`, i.e. 306 sensors sampled at 120 Hz over a 0.5 s window. The label is which of the four stimuli caused it. The model is **EEGNet** ([Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c)) — a deliberately tiny conv net (**6,100 parameters**) that learns *which frequencies* and *which sensors* matter.

> In one line: **read half a second of someone's brain and guess what they just heard or saw.**

(One subject, one session — so everything here is *within-person* decoding on a single recording, not a universal brain reader.)

## 1️⃣ A first result, and a sanity check

The first training run landed at **65.5% test accuracy**. Is that real? A 4-class problem has a 25% chance baseline, so I evaluated the **untrained** (randomly initialized) model:

| Model | Test accuracy |
|-------|--------------|
| Untrained (random weights) | **24.1%** (≈ chance) |
| Trained EEGNet | **65.5%** |

24.1% sitting right on the 25% chance line is exactly what you want to see — it confirms the input pipeline isn't leaking the label, so the trained model's jump to 65.5% is real learning, not a trivial artifact. So far so good.

## 2️⃣ Trap 1 — the split wasn't stratified

The pipeline did a random 60/20/20 train/val/test split with no class balancing. On a small dataset that produces lopsided splits:

| split | code 0 | code 1 | code 2 | code 3 |
|-------|:------:|:------:|:------:|:------:|
| test  | 17 | **9** | 18 | 14 |
| train | 35 | **56** | 41 | 41 |
| val   | 20 | **8** | 14 | 15 |

Class 1 got **56 of its 73 trials into training** but only 9 into test, while class 0 was barely trained. The dataset is essentially balanced (≈72/class) — the imbalance is purely a small-sample split artifact. The one-line fix:

```python
default_config["data"]["study"][1]["stratify_by"] = "code"
```

…which rebalances every split to ~14–15 per class in test.

## 3️⃣ Trap 2 — comparing two single runs lied to me

With stratification on, I retrained and compared:

| Split | Test accuracy |
|-------|--------------|
| Unstratified | **86.2%** |
| Stratified | **81.0%** |

So… *unstratified is better?!* That conclusion is wrong, for three compounding reasons:

- **Different test sets.** The two runs hold out different trials, so the numbers aren't comparable. With only 58 test samples, **1 trial = 1.7%** — the 5-point gap is 3 trials.
- **Tiny test set.** A 58-sample accuracy has a 95% interval of roughly ±9 points. 81% and 86% are statistically indistinguishable.
- **Checkpoint resume.** This was the real killer: the *same* unstratified config had scored 65.5% earlier and 86.2% now. Identical code, deterministic seed — but every run loaded and overwrote the same `last.ckpt`, so later runs silently **resumed training** and accumulated epochs. The 86.2% was a model that had trained far longer than I thought, scored on a lucky test set.

Lesson: **never compare single runs**, and train each model from scratch (`load_checkpoint=False`, `save_checkpoints=False`).

## 4️⃣ Doing it right — controlled, multi-seed

To isolate the *real* effect of stratification, I trained from scratch over many split seeds and compared the **distributions** instead of two lucky draws. Over 20 seeds:

| Split | mean | std | min | max |
|-------|:----:|:---:|:---:|:---:|
| stratified | 0.663 | **0.049** | 0.552 | 0.759 |
| unstratified | 0.651 | **0.066** | 0.517 | 0.810 |

Two findings:

1. **The means are tied** (0.663 vs 0.651) — stratification does *not* raise accuracy.
2. **It tightens the spread** and clips the worst-case tail (unstratified swings 0.517 → 0.810).

There was even a meta-lesson here: at *5* seeds the spread gap (std) looked like a dramatic **4×**; at 20 seeds it shrank to ~**1.35×** (0.066 vs 0.049). The dramatic version was a small-sample mirage — you need enough seeds to estimate variability itself.

So stratification is **free insurance against bad-luck splits**, not a performance lever.

## 5️⃣ The real ceiling — test-set size

Here's the deeper point hiding under all of this: the test set is only ~58 trials. Scoring a model on 58 examples is like polling 58 people — even with a perfect sample, the result can swing by roughly **±12%** (95%) just from *which* trials happened to land in the test set. That's the same kind of margin of error a 58-person political poll carries, and stratification can't touch it: balancing the classes doesn't make the test set any bigger. The only real fix is to **stop betting the whole answer on a single 58-trial draw**.

## 6️⃣ The capstone — repeated stratified k-fold

The honest evaluation: **repeated stratified 10-fold cross-validation**, every trial tested exactly once per pass, predictions pooled over all 288 trials, repeated 5× to check reproducibility, each model trained from scratch with an inner val split for early stopping.

```
Repeated stratified 10-fold × 5:  accuracy = 0.709,  honest 95% CI = ±0.053 (from 288 trials)
(the ±0.016 spread across the 5 repeats is reproducibility, NOT generalization certainty)
fold-level:                       0.709 ± 0.074 (std over 50 folds)
```

**70.9%, with an honest 95% CI of about ±5%** — and the numbers above *are* the whole lesson. A single ~29-trial fold bounces with std **0.074** (the small-sample lottery, alive and well). Testing on all 288 trials at once tightens the 95% interval to **±0.05** — about **2× tighter** than the 58-trial split's ±0.12, because you're finally scoring on every trial. The **±0.016** spread *across* the 5 repeats looks tighter still, but it's a trap: the repeats reuse the same 288 trials, so it measures how *reproducible* the procedure is, not how well you actually know the accuracy. The data floor is ±5%:

<svg viewBox="0 0 640 200" role="img" aria-label="Test-set size sets precision: a single 58-trial test has a ±0.12 (95%) interval; pooling all 288 trials tightens it to ±0.05, both centered on 0.709" style="width:100%;height:auto;max-width:640px;display:block;margin:1rem 0">
  <rect x="0.5" y="0.5" width="639" height="199" rx="10" fill="#ffffff" stroke="#e2e8f0"/>
  <line x1="180" y1="40" x2="180" y2="165" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="180" y1="165" x2="620" y2="165" stroke="#cbd5e1" stroke-width="1"/>
  <g font-size="12" fill="#64748b" text-anchor="middle">
    <text x="253" y="183">0.60</text>
    <text x="400" y="183">0.70</text>
    <text x="547" y="183">0.80</text>
  </g>
  <g font-size="13" fill="#334155" text-anchor="end">
    <text x="168" y="74">single 58-trial test</text>
    <text x="168" y="134">k-fold pooled (288)</text>
  </g>
  <!-- single 58-trial test, 95% CI ±0.12: 0.589..0.829 -->
  <line x1="237" y1="70" x2="590" y2="70" stroke="#f59e0b" stroke-width="3"/>
  <line x1="237" y1="62" x2="237" y2="78" stroke="#f59e0b" stroke-width="3"/>
  <line x1="590" y1="62" x2="590" y2="78" stroke="#f59e0b" stroke-width="3"/>
  <circle cx="413" cy="70" r="5" fill="#f59e0b"/>
  <text x="600" y="74" font-size="12" fill="#64748b">±0.12</text>
  <!-- k-fold pooled over 288, 95% CI ±0.05: 0.659..0.759 -->
  <line x1="340" y1="130" x2="487" y2="130" stroke="#10b981" stroke-width="3"/>
  <line x1="340" y1="122" x2="340" y2="138" stroke="#10b981" stroke-width="3"/>
  <line x1="487" y1="122" x2="487" y2="138" stroke="#10b981" stroke-width="3"/>
  <circle cx="413" cy="130" r="5" fill="#10b981"/>
  <text x="497" y="134" font-size="12" fill="#64748b">±0.05</text>
</svg>

For context, the full journey of the number:

| Stage | Reported | What it actually was |
|-------|:--------:|----------------------|
| First run | 65.5% | one noisy draw |
| "Best" run | 86.2% | inflated by checkpoint resume + a lucky imbalanced test set |
| **Honest (k-fold + CI)** | **71% ± 5%** | honest 95% CI, every trial tested |

Notice the k-fold mean (0.709) is even a bit *higher* than the multi-seed sweep (0.66) — because 10-fold trains each model on ~90% of the data instead of 60%, so it's less pessimistic. More training data, same center, and an honest ±5% error bar set by the 288 test trials — tighter than the 58-trial split's ±12%, but no tighter than the data allows.

## 7️⃣ Where the model actually wins

The flat 71% hides a **bimodal story**. Pooling out-of-fold predictions across all 5 repeats (288 trials × 5 = **1,440 predictions**) into a confusion matrix (rows = true, columns = predicted):

| true ↓ / pred → | Aud/L | Aud/R | Vis/L | Vis/R | per-class |
|---|:--:|:--:|:--:|:--:|:--:|
| **Aud/L** | 204 | 139 | 9 | 8 | **57%** |
| **Aud/R** | 131 | 211 | 12 | 11 | **58%** |
| **Vis/L** | 17 | 8 | 308 | 32 | **84%** |
| **Vis/R** | 8 | 13 | 31 | 298 | **85%** |

<svg viewBox="0 0 640 340" role="img" aria-label="Per-class accuracy: Aud/L 57%, Aud/R 58%, Vis/L 84%, Vis/R 85%" style="width:100%;height:auto;max-width:640px;display:block;margin:1rem 0">
  <rect x="0.5" y="0.5" width="639" height="339" rx="10" fill="#ffffff" stroke="#e2e8f0"/>
  <!-- axes -->
  <line x1="60" y1="30" x2="60" y2="280" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="60" y1="280" x2="620" y2="280" stroke="#cbd5e1" stroke-width="1"/>
  <!-- reference lines (short tags sit in the left margin, clear of the bars) -->
  <line x1="60" y1="217.5" x2="620" y2="217.5" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 4"/>
  <line x1="60" y1="155" x2="620" y2="155" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 4"/>
  <g font-size="11" fill="#64748b" text-anchor="end">
    <text x="55" y="221">25%</text>
    <text x="55" y="159">50%</text>
  </g>
  <!-- bars: auditory indigo, visual emerald -->
  <rect x="90"  y="138.25" width="80" height="141.75" fill="#6366f1"/>
  <rect x="230" y="135.5"  width="80" height="144.5"  fill="#6366f1"/>
  <rect x="370" y="69"     width="80" height="211"    fill="#10b981"/>
  <rect x="510" y="67.25"  width="80" height="212.75" fill="#10b981"/>
  <!-- value labels -->
  <g font-size="14" fill="#1e293b" text-anchor="middle" font-weight="bold">
    <text x="130" y="130">57%</text>
    <text x="270" y="127">58%</text>
    <text x="410" y="61">84%</text>
    <text x="550" y="59">85%</text>
  </g>
  <!-- x labels -->
  <g font-size="13" fill="#334155" text-anchor="middle">
    <text x="130" y="298">Aud/L</text>
    <text x="270" y="298">Aud/R</text>
    <text x="410" y="298">Vis/L</text>
    <text x="550" y="298">Vis/R</text>
  </g>
  <!-- group labels -->
  <g font-size="13" fill="#64748b" text-anchor="middle">
    <text x="200" y="322">auditory — which ear?</text>
    <text x="480" y="322">visual — which side?</text>
  </g>
</svg>

_Dashed lines mark chance level: **25%** for the full 4-class guess, **50%** for telling left from right within one modality._

The model is doing three very different things:

- **Sound vs. sight: ~94%** — cross-modality confusions are tiny. It nails *what kind* of stimulus.
- **Visual left vs. right: ~85%** — cleanly separated.
- **Auditory left vs. right: ~57%** (≈61% once you condition on getting the modality right) — reliably above the 50% coin-flip across hundreds of trials, but far weaker than vision. It only *weakly* recovers which ear a tone came from.

And that lines up with the neuroscience. The visual field is strongly contralateral and retinotopic — a left-field stimulus lights up right occipital cortex with a clear spatial signature. Auditory "which ear" is processed bilaterally with much weaker lateralization in MEG, so it's genuinely harder to decode. The signal clearly still carries *some* ear information — the model stays reliably above chance — but far less than the visual field offers, and on ~70 trials per class I can't cleanly separate "weak signal" from "tiny model starved of data." Either way, the decoder's error pattern recovers a known asymmetry of the brain, and that one weakness is what drags the ~71% down.

## 🎯 Takeaways

The model was the easy part. The evaluation was where all the learning happened:

1. **Always stratify** classification splits — free insurance against bad-luck imbalance. It won't raise your mean; it clips the worst case.
2. **Train from scratch** when comparing. Shared checkpoints silently resume and inflate results — the single most dangerous bug here (65.5% → 86.2% from nothing but resume).
3. **Never trust a single split.** Compare distributions over seeds, not two numbers.
4. **Test-set size caps your precision.** A 58-trial test carries a ~±12% margin (95%) no matter what; pooling k-fold over all 288 trials tightens it to ~±5%, but no cleverer split can — and averaging repeats only buys reproducibility, not certainty.
5. **Report mean ± CI**, and **always look at the per-class breakdown** — a flat accuracy hid the fact that 3 of 4 classes were strong and one was near chance.

The difference between a *number* and an *understanding*: I started with "65 vs 81 vs 86 — which is real?" and ended with "**71% ± 5%, and here is precisely where the model wins and loses.**" 🌱
