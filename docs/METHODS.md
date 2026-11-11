# Methods and interpretation rules

This document follows **Growth stage for Strawberry Nutrient Deficiency**, supplied on 11 November 2026. The source report determines the research question, two-run chronology and interpretation. Existing full-precision exports supply the numerical results.

## Question and scope

Does known growth stage improve prediction of tip burn versus yellowing treatment-group labels beyond a fixed image representation? The test concerns held-out plants from one available farm, not external farms, lesion diagnosis or tissue-confirmed nutrient deficiency.

The two proposed contributions are **prevalence shift** (additive stage information) and **expression shift** (stage–image interactions). Both classes occur at all five observed stages.

## Dataset

The report's full-release audit contains 172,037 labelled images from 12 farms and five classes. Tip burn and yellowing occur only at AI22_003; blight and wilt occur on three other farms without healthy plants. This confounding motivates restricting the ablation to the two nutrient-treatment labels at AI22_003.

Both runs use the source **validation split only**, approximately 6 GB. It includes all 55 plants from that farm: 33 tip burn and 22 yellowing. Adding training-split images would not introduce new plants.

The code reads `images.growth_stage`, `disease_class`, `farm_id`, `crops_id` and `fname`. It normalises `출뢰` to `출뢰기`, matches normalised image filenames, filters to AI22_003 and the two labels, and excludes empty stages or missing images. Report counts: 3,540 labels read; 37 excluded; 3,503 usable whole images (1,989 tip burn, 1,514 yellowing).

| Stage | English |
| --- | --- |
| 정식기 | Transplanting |
| 출뢰기 | Flower-bud emergence |
| 개화기 | Flowering |
| 과실비대기 | Fruit enlargement |
| 수확기 | Harvest |

The ordering also recognises `생육기` (vegetative growth), which is absent from this sample. The five observed dataset stages are not the seven Darlan classes; the report gives only approximate correspondences to Darlan and BBCH categories.

Run 2 retains images whose first category is `잎` (leaf), with a first-annotation box and positive image dimensions: 1,902 images (1,064 tip burn, 838 yellowing), still all 55 plants. The square crop is centred on that box and has side length 1.5 times the longer box dimension. The report's 50% margin is implemented as `pad=0.5`, not 50% on every side. Orientation and scale are adjusted using supplied dimensions; out-of-frame crop regions may be padded.

## Feature extraction and ablation arms

Frozen `convnext_tiny.fb_in22k` features have 768 dimensions. No backbone fine-tuning is performed. In each training fold, fit standardisation → 64-component PCA → standardisation, then apply those fitted objects to held-out plants.

Let Z contain the 64 image features and S the five one-hot stage indicators:

| Arm | Design matrix | Dimensions |
| --- | --- | ---: |
| Stage only | S | 5 |
| Image only | Z | 64 |
| Image + stage | Z, S | 69 |
| Image × stage | Z, S, Z×S1, …, Z×S5 | 389 |

All classifiers use logistic regression with C=0.5, balanced class weights and max_iter=5000. Hyperparameters are fixed across arms; input dimensionality differs. Everything runs on CPU.

Shuffled controls evaluate the fitted additive and interaction models after shuffling stage rows within the test fold. They do not retrain models and do not shuffle target labels. The same shuffled assignment is used for both stage arms in a fold. This diagnostic does not preserve individual plants' stage trajectories.

## Plant-grouped validation

The grouping key is farm ID + plant ID. Five-fold StratifiedGroupKFold is repeated five times with seeds 0–4. A plant cannot appear in training and test within the same fold. Preprocessing is fitted only on training rows.

Each image receives one out-of-fold probability per repeat. Average those five probabilities, then threshold at 0.5. Reported macro-F1 is calculated from the resulting predictions, not averaged across 25 fold scores.

## Bootstrap and intervals

Draw 55 plant IDs with replacement, include all their images, and repeat 2,000 times (seed 0). Each draw is shared by the arms so differences are paired. Predictions stay fixed; models are not refitted.

Scores remain image-level macro-F1. Cluster resampling does not make each plant contribute equally: plants with more photos contribute more observations. The original custom metric omits a true class absent from a resample; this behaviour remains for numerical continuity and can matter in sparse stage subsets.

| Quantity | Interval |
| --- | --- |
| Overall macro-F1 | 95% |
| Each stage arm versus image only | 97.5% |
| Correct versus shuffled stage | 95% |
| Stage-specific differences | 95%, exploratory |

The two main comparisons use 1 − 0.05/2 = 0.975 intervals, a Bonferroni adjustment within each run. This does not adjust across both runs or all subgroup/shuffle comparisons.

## Decision rules and chronology

1. **Set before run 1:** practical gain threshold Δ macro-F1 = 0.01.
2. **Added after run 1, before run 2:** interpret stage comparisons only when the image-only baseline's **lower 95% CI bound >0.60**.

The report says each rule was fixed before the run to which it applies. This is not a claim of external preregistration.

**The baseline gate takes precedence.** Run 2 has lower bound 0.522512…; stage comparisons and stage-specific contrasts remain available as numbers but carry `not_interpreted` status. They must not drive a keep/drop decision. Run 1 is described as inconclusive from its paired intervals; the later gate is not represented as prespecified for run 1.

The gate is a study-specific baseline-adequacy screen. It need not be a chance test to govern interpretation. A 0.5 reference line is illustrative; no calibrated null distribution or permutation chance test was supplied.

The shared `interpretation.py` applies the gate before interval verdicts. For a run eligible for interpretation, a lower difference bound >0.01 supports a gain above the practical margin; a positive bound below that does not establish a practically meaningful gain; an interval inside ±0.01 supports practical equivalence. The published run 1 intervals are inconclusive. These outputs do not claim a universal decision for all stage-conditioned architectures.

## Why the baseline is inadequate

Following the report's annotation interpretation, 3,281 images have no visible symptom (status N), 202 are symptomatic (Y), and 20 have E. E is kept separate; no new biological interpretation is assigned. Only 54.3% of images are leaves. The annotated box occupies a median of about 1.4% of the full frame.

The labels describe a plant's treatment group over time, including frames before visible onset. They are not interchangeable with each image's symptom annotation. Stage and time since treatment are entangled.

Run 2 changes both framing and inclusion of images, so the rise from 0.556 to 0.569 does not isolate cropping. Both runs use frozen features and one farm; conclusions do not extend to fine-tuned models, other cultivars or external farms.

## Reproduction and provenance

Use only the extracted validation subset for report reproduction. The portable script explicitly filters the farm and classes, rejects missing group IDs, duplicate labelled filenames, mixed labels within a plant and unknown stages, and checks train/test plant separation. Input audit JSON files compare aggregate counts with the report; matching counts do not prove source-split identity.

New feature caches have ordered metadata/file-state fingerprints. Existing feature arrays require exact row alignment; full-image data.csv is checked when supplied. Legacy crop caches have no independent manifest. The notebook retains its historical row-count cache checks; prefer the script for safer fresh extraction.

All notebook code outputs were cleared in this revision. Reported results remain in the aggregate CSVs; the original executed notebook and figures remain available in Git history. No feature extraction, full-data cross-validation or bootstrap was rerun during the report-alignment update. See VERIFICATION.md for actual checks.

## Follow-up

The report proposes a post-onset subset (about 15 plants per class with recorded onset), fine-tuning on leaf crops when GPU resources return, and repeating the same ablation if the baseline passes. Later frames still require symptom checks. The onset script mentioned in the report was not included in the source repository, so this revision does not claim an implemented or executed onset experiment.
