"""Builds and pushes a Kaggle script kernel that runs experiment queues.

The kernel installs this repository at a fixed commit, so every run is tied to the code
that produced it. On a "T4 x2" machine each queue runs on its own GPU in parallel.

Usage:
  python kernels/launch.py --slug hw2-frcnn --queue "python -m hw2.tv_train ..." --queue "..."
  python kernels/launch.py --slug hw2-prep --cpu --no-prep --queue "python -m hw2.prep --out {WORK}/prep --step 2"
Placeholders in commands: {PREP} (prepared dataset dir), {OUT} (runs dir), {WORK} (/kaggle/working).
Commands inside one --queue are separated by " && ".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

USER = "lizabugirrr"
REPO = "https://github.com/lizabugirrr/hw2_ai_bugir_pasichnyk"
COMP = "multiple-object-video-detection-hw-2-ucu-ai-2026"

RUNNER = r'''
import glob, os, subprocess, threading, time
SHA = {sha!r}
QUEUES = {queues!r}
WORK = "/kaggle/working"
t0 = time.time()
subprocess.run(
    "pip install -q ultralytics==8.4.191 pyarrow pycocotools && "
    f"pip install -q --no-deps git+{repo}@{{SHA}}", shell=True, check=True)
print(f"installed hw2@{{SHA}} in {{time.time()-t0:.0f}}s", flush=True)

found = []
for d, sub, files in os.walk("/kaggle/input"):
    sub[:] = [s for s in sub if s not in ("images", "labels", "ignore", "train", "val", "test", "runs")]
    if d.count("/") <= 4:
        print("input dir:", d, flush=True)
    if "frames.csv" in files:
        found.append(d)
PREP = sorted(found)[0] if found else ""
OUT = f"{{WORK}}/runs"
os.makedirs(f"{{WORK}}/logs", exist_ok=True)
print("PREP =", PREP, flush=True)

def run_queue(gpu, cmds):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS="ignore")
    wd = f"/tmp/work{{gpu}}"
    os.makedirs(wd, exist_ok=True)
    for i, cmd in enumerate(cmds):
        cmd = cmd.replace("{{PREP}}", PREP).replace("{{OUT}}", OUT).replace("{{WORK}}", WORK)
        log = f"{{WORK}}/logs/q{{gpu}}_{{i}}.log"
        print(f"[gpu{{gpu}}] start {{cmd}}", flush=True)
        s = time.time()
        p = subprocess.Popen(cmd, shell=True, cwd=wd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace")
        with open(log, "w") as f:
            for line in p.stdout:
                f.write(line)
                # echo compact lines (epoch summaries, metrics, errors), skip progress bars
                if len(line) < 600 and "\r" not in line and "━" not in line and "%|" not in line:
                    print(f"[gpu{{gpu}}] {{line}}", end="", flush=True)
        p.wait()
        print(f"[gpu{{gpu}}] exit {{p.returncode}} after {{(time.time()-s)/60:.1f}} min: {{cmd}}", flush=True)

threads = [threading.Thread(target=run_queue, args=(g, q)) for g, q in enumerate(QUEUES)]
[t.start() for t in threads]
[t.join() for t in threads]
print(f"all queues finished in {{(time.time()-t0)/60:.1f}} min", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--queue", action="append", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-prep", action="store_true", help="do not attach the hw2-prep kernel output")
    ap.add_argument("--sources", nargs="*", default=[], help="extra kernel outputs to attach")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    pushed = subprocess.run(["git", "branch", "-r", "--contains", sha], capture_output=True, text=True).stdout
    if not pushed.strip() and not a.dry:
        raise SystemExit(f"commit {sha} is not pushed; the kernel installs the package from GitHub")

    queues = [q.split(" && ") for q in a.queue]
    meta = {
        "id": f"{USER}/{a.slug}", "title": a.slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": not a.cpu, "enable_internet": True,
        "competition_sources": [COMP], "dataset_sources": [],
        "kernel_sources": ([] if a.no_prep else [f"{USER}/hw2-prep"]) + [f"{USER}/{s}" for s in a.sources],
    }
    if not a.cpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    d = Path(tempfile.mkdtemp(prefix=f"{a.slug}-"))
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    (d / "run.py").write_text(RUNNER.format(sha=sha, queues=queues, repo=REPO))
    print(f"kernel dir {d} (hw2@{sha[:7]})")
    if not a.dry:
        cmd = ["kaggle", "kernels", "push", "-p", str(d)] + ([] if a.cpu else ["--accelerator", "NvidiaTeslaT4"])
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
