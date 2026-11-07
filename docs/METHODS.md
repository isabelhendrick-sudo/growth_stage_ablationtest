# Methods and interpretation

## Research question

Does known growth stage improve prediction of the dataset's tipburn/yellowing class label beyond a fixed image-feature representation? The study evaluates new plants from the same available farm. It does not estimate disease causation, diagnose individual visible lesions, or establish performance at a new farm.

## Inputs and inclusion

The notebook reads `images.growth_stage`, `disease_class`, `farm_id`, `crops_id` and `fname` from JSON labels. It normalises `출뢰` to `출뢰기`, matches images by normalised filename, and keeps the two requested labels with non-empty stage and a matching image. This yields 3,503 images from 3,540 labels read. The group key is `farm_id + '_' + crops_id`.

The selected sample has 55 groups from one farm: 33 tipburn groups and 22 yellowing groups. No plant group has both class labels in the working table. Five stages are present:

| Korean value | English description |
| --- | --- |
| 정식기 | Transplanting |
| 출뢰기 | Budding |
| 개화기 | Flowering |
| 과실비대기 | Fruit enlargement |
| 수확기 | Harvest |

`생육기` is listed in the code's stage order but is not present in the selected sample.

The crop experiment retains images whose first category is `잎` (leaf), with a first-annotation bounding box and usable dimensions. It uses the first annotation/category only. A square crop is centred on the box with side length **1.5 times** its longest dimension. Image orientation and scale are adjusted using the supplied dimensions. Crops extending outside the image can contain padding. The preview montage contains the first four leaf examples per class, not a random or representative diagnostic sample.

## Features and model arms

The frozen `convnext_tiny.fb_in22k` network produces 768 features per image. Its [official model card](https://huggingface.co/timm/convnext_tiny.fb_in22k) identifies ImageNet-22k pretraining and documents classifier-free embedding extraction. The experiment does not fine-tune the backbone.

Within each training fold, the pipeline fits a standard scaler, 64-component PCA and a second standard scaler. Test images are transformed using those fitted training-fold objects. Let `Z` be the resulting 64 image features and `S` the five-dimensional one-hot stage vector:

| Arm | Classifier inputs | Dimensions |
| --- | --- | ---: |
| `stage_only` | `S` | 5 |
| `no_stage` | `Z` | 64 |
| `stage_additive` | `[Z, S]` | 69 |
| `stage_interaction` | `[Z, S, Z*S_1, …, Z*S_5]` | 389 |

All arms use logistic regression with `C=0.5`, balanced class weights and `max_iter=5000`. This fixed regularisation setting is not separately tuned for the different input dimensions. Conclusions therefore concern these particular representations and classifier settings, not every possible use of stage information.

## Held-out predictions

The analysis uses five repeats of five-fold `StratifiedGroupKFold`, with repeat seeds 0–4. Groups are plants, so images from a plant cannot be in both training and test sets in the same fold. The splitter attempts class balance while preserving group separation, as described in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

Each image receives one held-out probability per repeat. The five probabilities are averaged and thresholded at 0.5. Reported F1 is computed from these final predictions; it is not the arithmetic average of 25 fold-level F1 scores.

The two shuffled evaluations use each trained stage model with stage rows randomly permuted within its test fold. The same shuffled stage assignment is used for the additive and interaction arms in that fold. Training data and fitted classifiers are unchanged. This is an input-perturbation diagnostic, not a label-permutation test of prediction above chance. It does not preserve stage trajectories within plants.

## Metric and uncertainty

Macro-F1 is the unweighted mean of the two class-specific F1 scores ([metric documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)). The notebook's custom function omits a class when that class is absent from the true labels of a resample; this can affect rare stage-specific bootstrap samples and is preserved in the script for reproduction.

The bootstrap draws 55 plant IDs with replacement, includes all images for each selected plant, and repeats this 2,000 times with seed 0. Each draw is shared across arms, making differences paired. Models are not retrained in bootstrap resamples. Intervals therefore describe variation from resampling observed plant groups conditional on the existing held-out predictions; they do not capture the full uncertainty of model fitting, data collection or new farms.

Scores remain image-level macro-F1. Plant resampling respects within-plant clustering, but does not make each plant contribute equally to the pooled score. Plants with more images contribute more observations.

| Result | Nominal interval level |
| --- | --- |
| Overall macro-F1 by arm | 95% |
| Additive vs no-stage; interaction vs no-stage | 97.5% each, for two main comparisons within an experiment |
| Correct stage vs shuffled stage | 95% |
| Within-stage differences | 95%, exploratory and unadjusted |

The main intervals use the notebook's `1 - 0.05 / 2 = 0.975` setting, a Bonferroni-style adjustment for two comparisons within each experiment. It is not an adjustment across both experiments and all stage-specific/shuffle comparisons.

## Practical margin and decision statements

The notebook sets `MIN_DELTA = 0.01` macro-F1 as a practical-difference margin. Its presence in the code does not establish formal preregistration. The distinction is:

- A lower difference bound above 0 supports a positive effect under this interval procedure.
- A lower bound above +0.01 supports an effect exceeding the chosen practical margin.
- An interval fully inside −0.01 to +0.01 supports the notebook's practical-equivalence interpretation.
- An interval spanning meaningful benefit and harm is inconclusive.

All four main intervals span zero and extend beyond both sides of the practical band. Neither benefit nor equivalence is established.

The original notebook's automated `KEEP` branch checks positivity of the main and shuffle lower bounds, not whether the main lower bound exceeds +0.01. The portable script preserves those numeric rules but clarifies the wording. For the supplied results, both experiments remain inconclusive.

The final notebook cell calls the baseline interpretable only if its lower interval bound exceeds 0.60. **0.60 is not a statistical chance benchmark for macro-F1.** Falling below that heuristic does not invalidate the paired stage comparisons; it calls for caution about baseline usefulness. Conversely, a modest F1 score should not be called “chance” without a defined null comparison. A [dummy classifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) or an appropriately grouped permutation procedure would provide a clearer reference.

The latest notebook also draws a line at 0.5 labelled “Chance”. That line is an illustrative reference, not an empirically calibrated null distribution for this class balance and prediction rule. The stored notebook and supplied figures are preserved and annotated. The standalone script labels regenerated charts “Reference (0.5)” and replaces the final heuristic message with the distinction above.

## Annotation and sampling limits

- The first annotation's status is `N` in 3,281 images, `Y` in 202 and `E` in 20. Their meanings should be confirmed against the provider's codebook; no healthy/symptomatic mapping is assumed here.
- Full frames include leaf, fruit, stem and flower-cluster categories. The full-image model may therefore use context beyond leaf appearance.
- The median first-annotation bounding-box area is about 1.4% of the frame in both classes. It is not necessarily the area of a symptomatic lesion.
- Full-image and leaf-crop experiments use different image sets. A causal claim that cropping improves performance requires a matched-image comparison.
- The leaf experiment was explored after the initial analysis and is treated as exploratory. A final evaluation on untouched data remains necessary for a confirmatory claim.
- Adding more images of the same plants is not equivalent to adding more independent plants or farms.

## Portable-script changes

The modelling, seeds, transformations, prediction averaging and bootstrap calculations follow the notebook. The script additionally:

- Accepts input, output, font and optional feature-cache paths as arguments.
- Uses already-extracted data and does not write to the source tree.
- Checks for missing group IDs, duplicate labelled filenames, unknown stages and train/test plant overlap.
- Fingerprints new feature caches using ordered input metadata, image size/modification time, model identifier and crop settings.
- Checks explicit legacy arrays for shape and finite values, and checks full-image row order against `data.csv` when available. Legacy crop arrays have no original row manifest, so their identity cannot be independently proven from their shape.
- Exports per-stage differences and out-of-fold probability arrays, as well as the original result tables.
- Clarifies the decision wording without changing the original numeric results.

The notebook retains its original extraction and row-count-only feature-cache logic. Run the script for the additional safeguards. Fingerprints are cache-consistency checks, not cryptographic verification of every image's contents or a guarantee across software versions.
