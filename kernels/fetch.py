"""Downloads a finished kernel's runs into ./runs and rebuilds results/experiments.csv.

Usage: python kernels/fetch.py hw2-frcnn [hw2-retina ...]
       python kernels/fetch.py --table      (only rebuild the table)
Weights are skipped unless --weights is given (they are not tracked in git anyway).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

USER = "lizabugirrr"
ROOT = Path(__file__).resolve().parent.parent


def fetch(slug: str, weights: bool):
    tmp = Path(tempfile.mkdtemp(prefix=f"{slug}-"))
    cmd = ["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", str(tmp), "-o"]
    cmd += [] if weights else ["--file-pattern", r"^(?!.*\.pt$).*"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    for run in sorted((tmp / "runs").glob("*")):
        dst = ROOT / "runs" / run.name
        shutil.copytree(run, dst, dirs_exist_ok=True)
        (dst / "kernel.txt").write_text(slug + "\n")
        print("fetched", run.name)
    if (tmp / "logs").exists():
        shutil.copytree(tmp / "logs", ROOT / "runs" / "_logs" / slug, dirs_exist_ok=True)


def table() -> pd.DataFrame:
    rows = []
    for m in sorted((ROOT / "runs").glob("*/metrics.json")):
        cfg = json.loads((m.parent / "config.json").read_text())
        model = cfg.get("model")
        rows.append({"run": m.parent.name, "model": model, **json.loads(m.read_text())})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "experiments.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    for s in a.slugs:
        fetch(s, a.weights)
    df = table()
    cols = [c for c in ["run", "mAP", "AP50", "AP_small", "AP_medium", "AP_large", "AR500", "fps", "train_min"] if c in df]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
