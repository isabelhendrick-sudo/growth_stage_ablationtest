#!/usr/bin/env python3
"""Growth-stage ablation for strawberry tipburn and yellowing labels.

Uses already-extracted JSON labels and images. Recreates the original modelling
workflow and writes all new results below --out. See docs/METHODS.md for limits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path(os.environ.get('ABLATION_DATA_DIR', 'data')))
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    parser.add_argument('--feature-dir', type=Path, help='Optional existing features.npy and features_leafcrop.npy in EXACT dataset row order; see README')
    parser.add_argument('--font', default='Malgun Gothic', help='Installed font supporting Korean stage labels')
    return parser.parse_args()


def dataset_signature(data, mode):
    columns = ['fname', 'label', 'stage', 'group', 'image_path']
    if mode == 'leaf_crop':
        columns += ['bbox', 'W', 'H']
    rows = data[columns].to_dict('records')
    file_state = [(Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in data.image_path]
    payload = json.dumps({'rows': rows, 'files': file_state, 'model': 'convnext_tiny.fb_in22k',
                          'mode': mode, 'crop_pad': 0.5}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def extract_features(data, mode, output_dir, feature_dir, loader):
    import numpy as np
    if feature_dir is not None:
        name = 'features.npy' if mode == 'full_image' else 'features_leafcrop.npy'
        features = np.load(feature_dir / name, allow_pickle=False)
        if features.shape != (len(data), 768) or not np.isfinite(features).all():
            raise ValueError('Supplied features must be a finite (number_of_rows, 768) array.')
        if mode == 'full_image' and (feature_dir / 'data.csv').exists():
            import pandas as pd
            saved = pd.read_csv(feature_dir / 'data.csv')
            for column in ['fname', 'label', 'stage', 'group']:
                if list(saved[column]) != list(data[column]):
                    raise ValueError('Existing feature table does not match current row order: ' + column)
        print("Using supplied features; row identity is the caller's responsibility:", name, flush=True)
        return features
    signature = dataset_signature(data, mode)
    cache, manifest = output_dir / 'features.npy', output_dir / 'features.json'
    if cache.exists() and manifest.exists():
        if json.loads(manifest.read_text())['signature'] == signature:
            features = np.load(cache, allow_pickle=False)
            if features.shape == (len(data), 768) and np.isfinite(features).all():
                print('Loaded matching feature cache:', mode, flush=True)
                return features
    import torch
    import timm
    torch.set_num_threads(os.cpu_count() or 1)
    model = timm.create_model('convnext_tiny.fb_in22k', pretrained=True, num_classes=0).eval()
    tf = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(data), 32):
            images = [tf(loader(row)) for row in data.iloc[i:i + 32].itertuples()]
            chunks.append(model(torch.stack(images)).numpy())
            if i % 320 == 0:
                print(f'{mode}: {i:,}/{len(data):,}', flush=True)
    features = np.concatenate(chunks)
    np.save(cache, features)
    manifest.write_text(json.dumps({'signature': signature, 'mode': mode}, indent=2))
    return features


def run(args):
    # Source notebook cell 1
    import os, json, unicodedata
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = args.data_dir
    OUTPUT_ROOT = args.out
    OUT = OUTPUT_ROOT / "full_image"
    OUT.mkdir(parents=True, exist_ok=True)

    CLASSES = ("잎끝마름", "황화")
    STAGE_ORDER = ["정식기", "생육기", "출뢰기", "개화기", "과실비대기", "수확기"]
    MIN_DELTA = 0.01  # smallest macro-F1 gain worth keeping stage for - decide now, before seeing results

    plt.rcParams["font.family"] = args.font  # Korean text in plots on Windows
    plt.rcParams["axes.unicode_minus"] = False
    print("ROOT exists:", ROOT.exists())
    if not ROOT.is_dir():
        raise ValueError("Data directory not found. Supply --data-dir with extracted JSON labels and images.")

    # Source notebook cell 2
    #helper functions

    def nfc(s):
        return unicodedata.normalize("NFC", s)


    def read_json(p):
        for enc in ("utf-8", "utf-8-sig", "cp949"):
            try:
                with open(p, encoding=enc) as f:
                    return json.load(f)
            except UnicodeDecodeError:
                continue
            except (json.JSONDecodeError, OSError):
                return None
        return None

    # Source notebook cell 4
    #labels, image matching, and plant coverage
    KEYS = {"growth_stage": "stage", "disease_class": "label", "farm_id": "farm",
            "crops_id": "plant", "fname": "fname"}
    rows = []
    for jp in ROOT.rglob("*.json"):
        d = read_json(jp)
        if d is None:
            continue
        im = d.get("images", {})
        rows.append({new: str(im.get(old) or "").strip() for old, new in KEYS.items()})
    if not rows:
        raise ValueError("No readable JSON labels. Extract the source data before running.")
    df = pd.DataFrame(rows)
    df["stage"] = df.stage.replace({"출뢰": "출뢰기"})
    df["group"] = df.farm + "_" + df.plant  # one plant = one group

    img_paths = {nfc(fn).lower(): os.path.join(dp, fn)
                 for dp, _, fns in os.walk(ROOT) for fn in fns
                 if fn.lower().endswith((".jpg", ".jpeg", ".png"))}
    df["image_path"] = df.fname.map(lambda f: img_paths.get(nfc(f).lower()))

    data = df[df.label.isin(CLASSES) & (df.stage != "") & df.image_path.notna()].reset_index(drop=True)
    if data.empty:
        raise ValueError("No matching labelled images.")
    if data.group.eq("_").any() or (data.farm.eq("") | data.plant.eq("")).any():
        raise ValueError("Plant and farm IDs are required for grouped validation.")
    if data.fname.duplicated().any():
        raise ValueError("Duplicate labelled filenames; resolve duplicates before analysis.")
    stages = [s for s in STAGE_ORDER if s in set(data.stage)]
    data.to_csv(OUT / "data.csv", index=False, encoding="utf-8-sig")
    print(f"{len(df):,} labels read, {len(data):,} usable images, farms: {list(data.farm.unique())}")
    print(data.groupby("label").group.nunique().rename("plants").to_string())
    data.groupby(["stage", "label"]).group.nunique().unstack().reindex(stages)  # plants per stage x class
    if not set(data.stage).issubset(stages):
        raise ValueError("Unrecognised growth stage.")

    # Source notebook cell 5
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def load(p):
        with Image.open(p) as im:
            im.draft("RGB", (448, 448))
            return im.convert("RGB")

    X = extract_features(data, 'full_image', OUT, args.feature_dir, lambda row: load(row.image_path))
    print(X.shape, flush=True)

    # Source notebook cell 6
    #cross-validated ablation 
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    y = (data.label == CLASSES[1]).astype(int).values  # 0 = 잎끝마름, 1 = 황화
    s = data.stage.map({st: k for k, st in enumerate(stages)}).values
    S = np.eye(len(stages))[s]
    groups = data.group.values


    def design(Z, St, arm):
        if arm == "stage_only":
            return St
        if arm == "no_stage":
            return Z
        if arm == "stage_additive":
            return np.hstack([Z, St])
        return np.hstack([Z, St] + [Z * St[:, [k]] for k in range(St.shape[1])])  # stage_interaction


    ARMS = ["stage_only", "no_stage", "stage_additive", "stage_interaction"]
    ALL_ARMS = ARMS + ["stage_additive_shuffled", "stage_interaction_shuffled"]
    REPEATS, FOLDS, PCA_DIM = 5, 5, 64

    P = {a: np.zeros((REPEATS, len(y))) for a in ALL_ARMS}
    for r in range(REPEATS):
        rng = np.random.default_rng(r)
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=r)
        for tr, te in cv.split(X, y, groups):
            assert not set(groups[tr]) & set(groups[te]), "Plant leakage between folds"
            prep = make_pipeline(StandardScaler(), PCA(PCA_DIM, random_state=0), StandardScaler()).fit(X[tr])
            Ztr, Zte = prep.transform(X[tr]), prep.transform(X[te])
            Ste_shuf = S[te][rng.permutation(len(te))]  # stage scrambled at test time
            for arm in ARMS:
                clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")
                clf.fit(design(Ztr, S[tr], arm), y[tr])
                P[arm][r, te] = clf.predict_proba(design(Zte, S[te], arm))[:, 1]
                if arm in ("stage_additive", "stage_interaction"):
                    P[arm + "_shuffled"][r, te] = clf.predict_proba(design(Zte, Ste_shuf, arm))[:, 1]
        print(f"repeat {r + 1}/{REPEATS} done", flush=True)

    pred = {a: (P[a].mean(0) >= 0.5).astype(int) for a in ALL_ARMS}  # average over repeats
    np.savez_compressed(OUT / "oof_probabilities.npz", **P)

    # Source notebook cell 7
    #bootstrap by plant 
    def macro_f1(yt, yp):
        f = [2 * np.sum((yp == c) & (yt == c)) / (np.sum(yp == c) + np.sum(yt == c))
             for c in (0, 1) if np.any(yt == c)]
        return np.mean(f)


    ug, ginv = np.unique(groups, return_inverse=True)
    idx_by_plant = [np.where(ginv == g)[0] for g in range(len(ug))]
    rng = np.random.default_rng(0)
    B = 2000
    boot = {a: np.empty(B) for a in ALL_ARMS}
    boot_stage = {a: np.full((B, len(stages)), np.nan) for a in ALL_ARMS}
    for b in range(B):
        idx = np.concatenate([idx_by_plant[g] for g in rng.integers(0, len(ug), len(ug))])  # resample plants
        yb, sb = y[idx], s[idx]
        for a in ALL_ARMS:
            pb = pred[a][idx]
            boot[a][b] = macro_f1(yb, pb)
            for k in range(len(stages)):
                m = sb == k
                if m.any():
                    boot_stage[a][b, k] = macro_f1(yb[m], pb[m])

    # Source notebook cell 8
    #results and discussion 

    def ci(x, level=0.95):
        tail = (1 - level) / 2*100
        return np.nanpercentile(x, [tail, 100 - tail]) 

    def verdict(lo, hi, m=MIN_DELTA):
        if lo > m:
            return "HELPS"
        if lo > 0:
            return "helps, but maybe less than MIN_DELTA"
        if hi < -m:
            return "HURTS"
        if lo >= -m and hi <= m: 
            return "EQUIVALENT (no meaningful difference)"
        return "INCONCLUSIVE (CI too wide)"

    overall = pd.DataFrame([(a, macro_f1(y, pred[a]), *ci(boot[a])) for a in ALL_ARMS],
                           columns =["arm", "macro_f1", "ci_lo", "ci_hi"])

    MAIN = 1 - 0.05 / 2
    comparisons = [("stage_additive", "no_stage", MAIN), 
                   ("stage_interaction", "no_stage", MAIN), 
                   ("stage_additive", "stage_additive_shuffled", 0.95),
                   ("stage_interaction", "stage_interaction_shuffled", 0.95)]

    rows = []
    for a, b_, lev in comparisons:
        lo, hi = ci(boot[a] - boot[b_], lev)
        rows.append((f"{a} vs {b_}", macro_f1(y, pred[a]) - macro_f1(y, pred[b_]), lo, hi, verdict(lo, hi)))
    deltas = pd.DataFrame(rows, columns=["comparison", "delta", "ci_lo", "ci_hi", "verdict"])

    main, shuf = deltas.iloc[:2], deltas.iloc[2:]
    if ((main.ci_lo.values > 0) & (shuf.ci_lo.values > 0)).any():
        decision = "Positive gain and shuffle evidence under the original rule; check MIN_DELTA separately."
    elif (main.ci_hi <= MIN_DELTA).all():
        decision = f"DROP stage conditioning: confidently no gain larger than {MIN_DELTA} macro-F1."
    else:
        decision = "INCONCLUSIVE: intervals allow benefit and harm; more independent evidence is needed."

    overall.to_csv(OUT / "overall.csv", index=False)
    deltas.to_csv(OUT / "deltas.csv", index=False, encoding="utf-8-sig")
    print(overall.round(3).to_string(index=False))
    print(deltas.round(3).to_string(index=False))
    print("DECISION:", decision)
    (OUT / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    # Source notebook cell 9
    extra = []
    for jp in ROOT.rglob("*.json"):
        d = read_json(jp)
        if d is None:
            continue
        im = d.get("images", {})
        a = (d.get("annotations") or [{}])[0]
        c = (d.get("categories") or [{}])[0]
        bx = a.get("bbox") or [0, 0, 0, 0]
        W, H = float(im.get("width") or 0), float(im.get("height") or 0)
        extra.append({"fname": str(im.get("fname") or "").strip(),
                      "status": a.get("disease_status", ""),
                      "part": c.get("name", ""),
                      "bbox_frac": bx[2] * bx[3] / (W * H) if W and H else np.nan})

    d2 = data.merge(pd.DataFrame(extra), on="fname", how="left")
    pd.crosstab(d2.status, d2.label, margins=True).to_csv(OUT / "status_counts.csv")
    pd.crosstab(d2.part, d2.label).to_csv(OUT / "part_counts.csv")
    d2.groupby("label").bbox_frac.describe().to_csv(OUT / "bbox_summary.csv")

    for st in ("Y", "N"):
        m = (d2.status == st).values
        if m.any():
            print(f"no_stage macro-F1 on status={st}: {macro_f1(y[m], pred['no_stage'][m]):.3f}  (n={m.sum():,})")

    # Source notebook cell 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    o = overall.iloc[::-1]
    ax1.errorbar(o.macro_f1, range(len(o)), xerr=[o.macro_f1 - o.ci_lo, o.ci_hi - o.macro_f1],
                 fmt="o", capsize=3)
    ax1.set_yticks(range(len(o)), o.arm)
    ax1.set_xlabel("macro-F1 (95% CI, plants resampled)")
    ax1.set_title("Performance by arm")

    yy = np.arange(len(stages))
    ax2.axvspan(-MIN_DELTA, MIN_DELTA, color="0.9", label=f"±{MIN_DELTA} (no meaningful change)")
    ax2.axvline(0, color="0.4", lw=0.8)
    for off, a, col in ((-0.12, "stage_additive", "C0"), (0.12, "stage_interaction", "C1")):
        d = boot_stage[a] - boot_stage["no_stage"]
        point = [macro_f1(y[s == k], pred[a][s == k]) - macro_f1(y[s == k], pred["no_stage"][s == k])
                 for k in range(len(stages))]
        lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
        ax2.errorbar(point, yy + off, xerr=[np.array(point) - lo, hi - np.array(point)],
                     fmt="o", color=col, capsize=3, label=a)
    ax2.set_yticks(yy, stages)
    ax2.invert_yaxis()
    ax2.set_xlabel("macro-F1 change vs no_stage (95% CI)")
    ax2.set_title("Where does stage help?")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "ablation_results.png", dpi=160)
    plt.close(fig)
    stage_rows = []
    for a in ("stage_additive", "stage_interaction"):
        for k, stage in enumerate(stages):
            mask = s == k
            lo, hi = ci(boot_stage[a][:, k] - boot_stage["no_stage"][:, k])
            stage_rows.append({"arm": a, "stage": stage,
                               "delta": macro_f1(y[mask], pred[a][mask]) - macro_f1(y[mask], pred["no_stage"][mask]),
                               "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(stage_rows).to_csv(OUT / "stage_deltas.csv", index=False)


    # Source notebook cell 11
    #leaf images with their boxes 
    meta = []
    for jp in ROOT.rglob("*.json"):
        d = read_json(jp)
        if d is None:
            continue
        im = d.get("images", {})
        a = (d.get("annotations") or [{}])[0]
        c = (d.get("categories") or [{}])[0]
        meta.append({"fname": str(im.get("fname") or "").strip(),
                     "part": c.get("name", ""),
                     "bbox": a.get("bbox"),
                     "W": float(im.get("width") or 0),
                     "H": float(im.get("height") or 0)})
    meta = pd.DataFrame(meta)

    base = pd.read_csv((OUTPUT_ROOT / "full_image") / "data.csv")  # the original 3,503-image table
    leaf = base.merge(meta, on="fname", how="left")
    leaf = leaf[(leaf.part == "잎") & leaf.bbox.notna() & (leaf.W > 0)].reset_index(drop=True)
    leaf_stages = [s for s in STAGE_ORDER if s in set(leaf.stage)]

    print(f"{len(leaf):,} leaf images")
    print(leaf.groupby("label").group.nunique().rename("plants").to_string())
    leaf.groupby(["stage", "label"]).group.nunique().unstack().reindex(leaf_stages)
    if leaf.empty:
        raise ValueError("No matched leaf annotations.")

    # Source notebook cell 12
    #check the crops look okay 
    from PIL import ImageOps


    def load_crop(row, pad=0.5):
        im = Image.open(row.image_path)
        if im.size != (int(row.W), int(row.H)):
            im = ImageOps.exif_transpose(im)  # photo stored rotated
        sx, sy = im.size[0] / row.W, im.size[1] / row.H  # photo stored at a different size
        x, y, w, h = row.bbox
        cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
        side = max(w * sx, h * sy) * (1 + pad)
        return im.crop((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)).convert("RGB")


    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, r in zip(axes.flat, leaf.groupby("label").head(4).itertuples()):
        ax.imshow(load_crop(r))
        ax.set_title(f"{r.label} / {r.stage}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "leaf_crop_examples.png", dpi=160)
    plt.close(fig)

    # Source notebook cell 13
    crop_out = OUTPUT_ROOT / 'leaf_crop'
    crop_out.mkdir(parents=True, exist_ok=True)
    X_crop = extract_features(leaf, 'leaf_crop', crop_out, args.feature_dir, load_crop)
    print(X_crop.shape, flush=True)

    # Source notebook cell 14
    #switch the notebook to the crop data
    data, X, stages = leaf, X_crop, leaf_stages
    OUT = (OUTPUT_ROOT / "leaf_crop")  # keep the first run's results separate
    OUT.mkdir(exist_ok=True)
    print(len(data), X.shape, stages)
    data.to_csv(OUT / "data.csv", index=False, encoding="utf-8-sig")

    # Source notebook cell 15
    #cross-validated ablation 
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    y = (data.label == CLASSES[1]).astype(int).values  # 0 = 잎끝마름, 1 = 황화
    s = data.stage.map({st: k for k, st in enumerate(stages)}).values
    S = np.eye(len(stages))[s]
    groups = data.group.values


    def design(Z, St, arm):
        if arm == "stage_only":
            return St
        if arm == "no_stage":
            return Z
        if arm == "stage_additive":
            return np.hstack([Z, St])
        return np.hstack([Z, St] + [Z * St[:, [k]] for k in range(St.shape[1])])  # stage_interaction


    ARMS = ["stage_only", "no_stage", "stage_additive", "stage_interaction"]
    ALL_ARMS = ARMS + ["stage_additive_shuffled", "stage_interaction_shuffled"]
    REPEATS, FOLDS, PCA_DIM = 5, 5, 64

    P = {a: np.zeros((REPEATS, len(y))) for a in ALL_ARMS}
    for r in range(REPEATS):
        rng = np.random.default_rng(r)
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=r)
        for tr, te in cv.split(X, y, groups):
            assert not set(groups[tr]) & set(groups[te]), "Plant leakage between folds"
            prep = make_pipeline(StandardScaler(), PCA(PCA_DIM, random_state=0), StandardScaler()).fit(X[tr])
            Ztr, Zte = prep.transform(X[tr]), prep.transform(X[te])
            Ste_shuf = S[te][rng.permutation(len(te))]  # stage scrambled at test time
            for arm in ARMS:
                clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")
                clf.fit(design(Ztr, S[tr], arm), y[tr])
                P[arm][r, te] = clf.predict_proba(design(Zte, S[te], arm))[:, 1]
                if arm in ("stage_additive", "stage_interaction"):
                    P[arm + "_shuffled"][r, te] = clf.predict_proba(design(Zte, Ste_shuf, arm))[:, 1]
        print(f"repeat {r + 1}/{REPEATS} done", flush=True)

    pred = {a: (P[a].mean(0) >= 0.5).astype(int) for a in ALL_ARMS}  # average over repeats
    np.savez_compressed(OUT / "oof_probabilities.npz", **P)

    # Source notebook cell 16
    #bootstrap by plant 
    def macro_f1(yt, yp):
        f = [2 * np.sum((yp == c) & (yt == c)) / (np.sum(yp == c) + np.sum(yt == c))
             for c in (0, 1) if np.any(yt == c)]
        return np.mean(f)


    ug, ginv = np.unique(groups, return_inverse=True)
    idx_by_plant = [np.where(ginv == g)[0] for g in range(len(ug))]
    rng = np.random.default_rng(0)
    B = 2000
    boot = {a: np.empty(B) for a in ALL_ARMS}
    boot_stage = {a: np.full((B, len(stages)), np.nan) for a in ALL_ARMS}
    for b in range(B):
        idx = np.concatenate([idx_by_plant[g] for g in rng.integers(0, len(ug), len(ug))])  # resample plants
        yb, sb = y[idx], s[idx]
        for a in ALL_ARMS:
            pb = pred[a][idx]
            boot[a][b] = macro_f1(yb, pb)
            for k in range(len(stages)):
                m = sb == k
                if m.any():
                    boot_stage[a][b, k] = macro_f1(yb[m], pb[m])

    # Source notebook cell 17
    #results and discussion 

    def ci(x, level=0.95):
        tail = (1 - level) / 2*100
        return np.nanpercentile(x, [tail, 100 - tail]) 

    def verdict(lo, hi, m=MIN_DELTA):
        if lo > m:
            return "HELPS"
        if lo > 0:
            return "helps, but maybe less than MIN_DELTA"
        if hi < -m:
            return "HURTS"
        if lo >= -m and hi <= m: 
            return "EQUIVALENT (no meaningful difference)"
        return "INCONCLUSIVE (CI too wide)"

    overall = pd.DataFrame([(a, macro_f1(y, pred[a]), *ci(boot[a])) for a in ALL_ARMS],
                           columns =["arm", "macro_f1", "ci_lo", "ci_hi"])

    MAIN = 1 - 0.05 / 2
    comparisons = [("stage_additive", "no_stage", MAIN), 
                   ("stage_interaction", "no_stage", MAIN), 
                   ("stage_additive", "stage_additive_shuffled", 0.95),
                   ("stage_interaction", "stage_interaction_shuffled", 0.95)]

    rows = []
    for a, b_, lev in comparisons:
        lo, hi = ci(boot[a] - boot[b_], lev)
        rows.append((f"{a} vs {b_}", macro_f1(y, pred[a]) - macro_f1(y, pred[b_]), lo, hi, verdict(lo, hi)))
    deltas = pd.DataFrame(rows, columns=["comparison", "delta", "ci_lo", "ci_hi", "verdict"])

    main, shuf = deltas.iloc[:2], deltas.iloc[2:]
    if ((main.ci_lo.values > 0) & (shuf.ci_lo.values > 0)).any():
        decision = "Positive gain and shuffle evidence under the original rule; check MIN_DELTA separately."
    elif (main.ci_hi <= MIN_DELTA).all():
        decision = f"DROP stage conditioning: confidently no gain larger than {MIN_DELTA} macro-F1."
    else:
        decision = "INCONCLUSIVE: intervals allow benefit and harm; more independent evidence is needed."

    overall.to_csv(OUT / "overall.csv", index=False)
    deltas.to_csv(OUT / "deltas.csv", index=False, encoding="utf-8-sig")
    print(overall.round(3).to_string(index=False))
    print(deltas.round(3).to_string(index=False))
    print("DECISION:", decision)
    (OUT / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    # Source notebook cell 18
    # Readable names for the figures
    ARM_NAMES = {"stage_only": "Stage only", "no_stage": "Image only (no stage)",
                 "stage_additive": "Image + stage", "stage_interaction": "Image \u00d7 stage",
                 "stage_additive_shuffled": "Image + shuffled stage",
                 "stage_interaction_shuffled": "Image \u00d7 shuffled stage"}
    STAGE_NAMES = {"정식기": "Transplanting", "생육기": "Vegetative", "출뢰기": "Flower bud",
                   "개화기": "Flowering", "과실비대기": "Fruit enlargement", "수확기": "Harvest"}

    # Plain matplotlib look: framed panels, ticks pointing inwards
    PLAIN = {"font.size": 10, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
             "xtick.top": True, "ytick.right": True}


    def plot_by_input(overall, title, path):
        """Macro-F1 for each model input, with 95% confidence intervals."""
        with plt.rc_context(PLAIN):
            fig, ax = plt.subplots(figsize=(7.5, 5))
            o = overall.iloc[::-1].reset_index(drop=True)
            for i, r in o.iterrows():
                shuffled = r.arm.endswith("shuffled")
                ax.errorbar(r.macro_f1, i, xerr=[[r.macro_f1 - r.ci_lo], [r.ci_hi - r.macro_f1]], fmt="o",
                            color="0.5" if shuffled else "steelblue", mfc="white" if shuffled else "steelblue",
                            capsize=3, ms=6, lw=1.2)
            ax.axvline(0.5, color="0.4", ls="--", lw=0.8)
            ax.set_yticks(range(len(o)), [ARM_NAMES.get(a, a) for a in o.arm])
            ax.set_ylim(-0.6, len(o) - 0.4)
            ax.set_xlabel("Macro-F1 (95% CI, plants resampled)")
            ax.grid(axis="x", color="0.9", lw=0.5)
            handles = [plt.Line2D([], [], marker="o", color="steelblue", ls="-", label="Model input"),
                       plt.Line2D([], [], marker="o", color="0.5", mfc="white", ls="-",
                                  label="Control: stage shuffled at test time"),
                       plt.Line2D([], [], color="0.4", ls="--", lw=0.8, label="Reference (0.5)")]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2,
                      frameon=False, fontsize=9)
            ax.set_title(title, fontsize=13, fontweight="bold")
            fig.tight_layout()
            fig.savefig(path, dpi=300)
            plt.close(fig)


    def plot_by_stage(boot_stage, title, path):
        """Change in macro-F1 from adding stage, at each growth stage."""
        with plt.rc_context(PLAIN):
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            yy = np.arange(len(stages))
            ax.axvspan(-MIN_DELTA, MIN_DELTA, color="0.88", lw=0, label=f"\u00b1{MIN_DELTA} (no meaningful change)")
            ax.axvline(0, color="0.3", lw=0.8)
            for off, a, col in ((-0.14, "stage_additive", "steelblue"), (0.14, "stage_interaction", "darkorange")):
                d = boot_stage[a] - boot_stage["no_stage"]
                point = np.array([macro_f1(y[s == k], pred[a][s == k]) - macro_f1(y[s == k], pred["no_stage"][s == k])
                                  for k in range(len(stages))])
                lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
                ax.errorbar(point, yy + off, xerr=[point - lo, hi - point], fmt="o", color=col, capsize=3,
                            ms=6, lw=1.2, label=ARM_NAMES[a])
            n_img = [int((s == k).sum()) for k in range(len(stages))]
            ax.set_yticks(yy, [f"{STAGE_NAMES.get(st, st)}\nn = {n:,} images" for st, n in zip(stages, n_img)])
            ax.invert_yaxis()
            ax.set_xlabel("Change in macro-F1 vs image only (95% CI)")
            ax.grid(axis="x", color="0.9", lw=0.5)
            ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3, frameon=False, fontsize=9)
            ax.set_title(title, fontsize=13, fontweight="bold")
            fig.tight_layout()
            fig.savefig(path, dpi=300)
            plt.close(fig)
    stage_rows = []
    for a in ("stage_additive", "stage_interaction"):
        for k, stage in enumerate(stages):
            mask = s == k
            lo, hi = ci(boot_stage[a][:, k] - boot_stage["no_stage"][:, k])
            stage_rows.append({"arm": a, "stage": stage,
                               "delta": macro_f1(y[mask], pred[a][mask]) - macro_f1(y[mask], pred["no_stage"][mask]),
                               "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(stage_rows).to_csv(OUT / "stage_deltas.csv", index=False)


    # Source notebook cell 19
    plot_by_input(overall, "Macro-F1 by model input | leaf crops", OUT / "ablation_by_input.png")
    plot_by_stage(boot_stage, "Effect of adding growth stage on macro-F1 | leaf crops", OUT / "ablation_by_stage.png")
    b = overall.set_index("arm").loc["no_stage"]
    print(f"Leaf-crop no-stage macro-F1: {b.macro_f1:.3f} (95% CI {b.ci_lo:.3f}-{b.ci_hi:.3f})")
    print("The notebook's 0.60 baseline screen is not a chance test. Interpret paired intervals and dataset limitations.")

if __name__ == "__main__":
    run(arguments())
