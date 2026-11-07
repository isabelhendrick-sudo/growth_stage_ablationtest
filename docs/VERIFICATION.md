# Verification record

Prepared from **`ablation_test (1).ipynb`**, the latest attachment, on **7 November 2026**.

- Compared the two notebook versions: the data preparation and statistical analysis cells are unchanged; the latest version adds separate English-labelled leaf-crop figures.
- Executed the portable data preparation, 5 × 5 grouped cross-validation, paired plant bootstrap and result export for both experiments using the available cached feature arrays.
- Reconstructed the original full-image data order and checked it against the saved working table.
- Verified 3,503 full images, 1,902 leaf images, 55 plant groups and one farm. No plant appears in both train and test sets within a fold.
- Compared all original overall-score and paired-difference values with the rerun: maximum absolute difference **0**.
- Executed the latest plotting helpers separately using the verified held-out predictions and regenerated bootstrap values. The stage chart matches the supplied high-resolution chart pixel-for-pixel: **true**. The input chart intentionally uses the clearer `Reference (0.5)` legend in newly generated output.
- Preserved the three source result PNG files byte-for-byte in `figures/`. Extracted the crop montage directly from the latest notebook.
- Validated the published notebook structure, all Python cells, the script's help command and local Markdown links.

**Scope:** feature extraction was not repeated and no weights were downloaded. The replay depends on the original 768-dimensional feature caches. Legacy crop arrays do not have an independent row manifest; reproducing the stored results does not establish their original feature-to-image provenance. The original notebook's stored output is retained and is not described as a newly executed notebook.

Raw source images, labels, feature arrays, prediction arrays and path-bearing working tables are excluded from the package. The published crop montage is already part of the supplied notebook. See `results/provenance.json` for hashes and local package versions.
