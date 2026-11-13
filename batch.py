#!/usr/bin/env python
"""Run a TeamTomo workflow on one, several or all tilt series, one after another on one GPU.

    python batch.py --config configs/xml_to_tomograms_warp_project.yaml               # every series in the project
    python batch.py --config configs/tsa_to_tomograms_warp_project.yaml --series 00254 # just one
    python batch.py --config my.yaml --overwrite --device cuda:1

Everything is decided by the config file (see configs/ and README.md); command-line flags only override it.
Results go to <output.root>/<series>/ (tomogram, run.json, ...); finished series are skipped on a re-run (resume);
a series that fails is logged and the batch continues. Summary: <output.root>/batch_summary.csv.
On a cluster, submit this one script as a single GPU job (no array needed).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teamtomo_workflows import load_config, run_batch  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", required=True, type=Path, help="YAML or JSON run configuration")
ap.add_argument("--series", nargs="+", default=None, help="only these series (default: input.series in the config)")
ap.add_argument("--overwrite", action="store_true", help="re-run series that already finished")
ap.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu (default: compute.device in the config)")
ap.add_argument("--stop_on_error", action="store_true")
a = ap.parse_args()

overrides = {}
if a.series:
    overrides.setdefault("input", {})["series"] = a.series
if a.overwrite:
    overrides.setdefault("output", {})["overwrite"] = True
if a.device:
    overrides["compute"] = {"device": a.device}
cfg = load_config(a.config, overrides)
rows = run_batch(cfg, stop_on_error=a.stop_on_error)
for r in rows:
    print(f"{r['series']:>12s}  {r['status']:<16s} {r.get('total_s') or '':>6}  {r.get('error', '')}")
print(f"-> {cfg['output']['root']}")
sys.exit(1 if any(r["status"] == "failed" for r in rows) else 0)
