"""Exercise both runs on small synthetic inputs; never a scientific reproduction."""
from pathlib import Path
import csv
import json
import subprocess
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "_qa" / "synthetic"
data_dir, features_dir, out_dir = QA/"data", QA/"features", QA/"outputs"
for directory in (data_dir, features_dir, out_dir):
    directory.mkdir(parents=True, exist_ok=True)
stages = ["정식기", "출뢰기", "개화기", "과실비대기", "수확기"]
classes = ["잎끝마름", "황화"]
for p in range(55):
    for t in range(2):
        k=p*2+t
        stem=f"{k:03d}"
        Image.new("RGB",(64,64),(100,150,100)).save(data_dir/(stem+".png"))
        record={"images":{"fname":stem+".png","farm_id":"AI22_003","crops_id":f"{p:03d}",
                          "growth_stage":stages[(p+t)%5],"disease_class":classes[int(p>=33)],
                          "width":64,"height":64},
                "annotations":[{"bbox":[16,16,24,24],"disease_status":"N"}],
                "categories":[{"name":"잎"}]}
        (data_dir/(stem+".json")).write_text(json.dumps(record,ensure_ascii=False),encoding="utf-8")
# A different farm must be excluded even when class/stage/image otherwise qualify.
other=json.loads((data_dir/"000.json").read_text(encoding="utf-8"))
other["images"]["farm_id"]="OTHER_FARM"
other["images"]["fname"]="other.png"
Image.new("RGB",(64,64),(100,150,100)).save(data_dir/"other.png")
(data_dir/"other.json").write_text(json.dumps(other,ensure_ascii=False),encoding="utf-8")
rng=np.random.default_rng(20261111)
features=rng.normal(size=(110,768)).astype("float32")
np.save(features_dir/"features.npy",features)
np.save(features_dir/"features_leafcrop.npy",features)
subprocess.run([sys.executable,str(ROOT/"ablation_test.py"),"--data-dir",str(data_dir),
                "--feature-dir",str(features_dir),"--out",str(out_dir)],check=True,cwd=ROOT)
for run in ("full_image","leaf_crop"):
    base=out_dir/run
    audit=json.loads((base/"input_audit.json").read_text())
    assert audit["observed"]["images"]==110
    assert audit["observed"]["plants"]==55
    assert not audit["report_counts_match"]
    summary=json.loads((base/"interpretation.json").read_text())
    with (base/"deltas.csv").open(encoding="utf-8-sig") as f:
        deltas=list(csv.DictReader(f))
    assert len(deltas)==4
    assert [float(r["ci_level"]) for r in deltas]==[0.975,0.975,0.95,0.95]
    if run=="leaf_crop":
        assert summary["stage_comparison_status"]=="not_interpreted", summary
        assert all(r["verdict"].startswith("NOT INTERPRETED") for r in deltas)
    else:
        assert not summary["baseline_gate_applies"]
    assert (base/"decision.txt").read_text(encoding="utf-8").strip()==summary["decision"]
    with np.load(base/"oof_probabilities.npz") as probabilities:
        assert all(probabilities[k].shape==(5,110) for k in probabilities.files)
with (out_dir/"leaf_crop"/"confusion_matrix.csv").open(encoding="utf-8") as f:
    cm=list(csv.DictReader(f))
assert sum(int(r[k]) for r in cm for k in ("predicted_tipburn","predicted_yellowing"))==110
print("Synthetic two-run smoke test passed: farm filter, audits, gated verdicts, intervals, OOF and confusion exports.")
