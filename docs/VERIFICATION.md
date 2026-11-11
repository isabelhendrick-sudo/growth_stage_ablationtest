# Verification record

## Report-alignment revision — 11 November 2026

Compared the repository at commit `36140b6edea305cd9d8b0d979c880b175e24dda1` with **Growth_stage_for_Strawberry_Nutrient_Deficiency.docx**, supplied on 11 November 2026.

### Changes

- Rewrote the README and methods around the report's motivation, single-farm restriction, whole-image run, annotation audit, leaf-crop run, decision-rule chronology and next steps.
- Made run 2's baseline gate govern generated decisions, paired-result verdicts and interpretation metadata. The requirement is strictly **lower 95% CI >0.60**.
- Kept run 1 descriptive, with its original inconclusive comparisons; did not retrospectively call the later gate prespecified for run 1.
- Added report Table 6's confusion matrix and Table 1's full-release dataset totals with explicit transcription provenance.
- Added two summary figures generated from existing aggregate CSVs and visually checked their rendered PNG previews.
- Filtered the portable analysis and notebook to AI22_003, added input-count audits to the script, and labelled follow-up experiments as unexecuted.
- Cleared all notebook code outputs and execution counts. The original executed notebook remains in Git history; none of the revised cells is represented as already executed.
- Kept original PNG figures and full-precision scores. Added interpretation/status metadata without changing scientific result values.

### Checks performed in this revision

- Compared **288 numeric fields** across every existing result CSV against the original repository: **all unchanged**.
- Checked report Table 2's plant coverage and both run image totals.
- Reconstructed the leaf-crop image-only macro-F1 from the reported confusion matrix; it agrees with the stored score to 14 decimal places.
- Passed **8 unit tests**, covering the strict threshold, gate precedence over apparently favourable stage differences, rule chronology, practical margin, published decisions, confusion matrix, coverage and notebook syntax/output state.
- Executed a **synthetic two-run integration test** with 110 artificial images, 55 artificial plants and random cached feature vectors. It completed both 5×5 cross-validation runs and both 2,000-resample bootstraps, excluded an additional wrong-farm record, wrote audit warnings for non-report counts, and verified gated verdicts, interval levels, probability shapes and confusion exports.
- Compiled the Python files, checked the command-line help and validated the notebook with nbformat.
- Checked local Markdown links against the final repository paths, including preserved original figures.

Commands:

```bash
python -m unittest discover -s tests -v
python tests/smoke_pipeline.py
python -m py_compile ablation_test.py interpretation.py render_report_figures.py
python ablation_test.py --help
python render_report_figures.py
```

The current checks used the available Python 3.14.6 environment. They do not constitute reproduction of the originally recorded Python 3.13.9 environment.

### What was not rerun

**No original AI Hub images, feature arrays or prediction arrays were available in this checkout.** Feature extraction, full-data model fitting and full-data bootstrap were not repeated. The synthetic test checks software behaviour only and supplies no research evidence. Report-only totals and confusion counts are transcriptions, not a fresh dataset audit.

The report mentions code for a post-onset experiment; it was absent from the original repository. This revision does not claim to have supplied or run it. The source report itself is not bundled.

## Inherited verification — 7 November 2026

The previous repository record described a replay from `ablation_test (1).ipynb` using cached feature arrays, matching original aggregate scores and differences with maximum absolute difference 0. It also recorded plotting checks and original figure hashes.

Those are **historical claims retained from the previous commit**, not checks repeated here. The top-level fields in [provenance.json](../results/provenance.json) retain that record; its `report_alignment` section describes this revision separately.

The inherited limitations remain: feature extraction was not repeated in that replay, and legacy leaf-crop feature arrays had no independent row manifest. Matching aggregate results alone does not establish original feature-to-image provenance.
