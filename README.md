# TeamTomo workflows

Two workflows, each as a notebook (look at one tilt series, with figures) and as a batch script (all tilt series,
for clusters and agents). Both run the same code in `teamtomo_workflows/`, built on [TeamTomo](https://github.com/teamtomo/teamtomo).

**Fully run examples** (EMPIAR-10499, series 00254) as web pages: see the Releases page of this repository — open
`XML_to_tomograms.html` or `TSA_to_tomograms.html` in a browser, no install needed.

| I have… | Workflow | Notebook | Typical time (00254, one GPU) |
| --- | --- | --- | --- |
| A Warp / M / MissAlignment XML (alignment done) | **XML → tomograms** | `XML_to_tomograms.ipynb` | ~10 s per series |
| Tilt images + tilt angles, no alignment | **TSA → tomograms** (tilt-series alignment) | `TSA_to_tomograms.ipynb` | ~1.5 min per series (CTF included) |

## Inputs
* **A Warp project** (most common): the folder that contains `warp_tiltseries/`, `warp_frameseries/`, `tomostar/` and
  `warp_tiltseries.settings`. XML → tomograms needs the XMLs in `warp_tiltseries/`; TSA → tomograms takes tilt angles
  and doses from `tomostar/`, images from `warp_tiltseries/tiltstack/`, and CTF from Warp's full-size averages.
* **Plain tilt stacks** (TSA → tomograms only): `<name>.st` with `<name>.rawtlt` (or `.tlt`) next to it.

## Run
```bash
python batch.py --config configs/xml_to_tomograms_warp_project.yaml               # every series in the project
python batch.py --config configs/tsa_to_tomograms_warp_project.yaml --series 00254 # one series
python batch.py --config configs/tsa_to_tomograms_plain_stacks.yaml                # plain stacks
```
Copy a config from `configs/`, change the paths, done. Every option is in the config (defaults and comments:
`teamtomo_workflows/config.py`); `--series`, `--overwrite` and `--device` override it. On a cluster, submit
`batch.py` as one GPU job: it runs the series one after another, skips series that already finished (so a re-run
resumes), and logs a failing series without stopping the others.

## Outputs (everything under one folder, default `teamtomo_output/`)
```
teamtomo_output/
  batch_summary.csv            one row per series per batch run (status, time, axis angle, thickness)
  logs/batch_<time>.log
  <series>/
    <series>_<px>Apx.mrc       tomogram
    run.json                   config used, package versions, GPU, timings, results, status
    alignment.json             (TSA) alignment as a TeamTomo TiltSeries, Angstrom
    ctf.json                   (TSA) defocus per tilt (when Warp's full-size averages exist)
    <series>.xml               (TSA, Warp project) alignment as a Warp XML -> continue in WarpTools / M
```

## What "Warp-compatible" means
TeamTomo's `reconstruct_tomogram` with Warp's conventions: true image pixel size, Warp's volume centre, Warp's
preprocessing, per-tilt CTF sign and dose weighting, and Warp's sampling weights. Given the same XML it reproduces
WarpTools `ts_reconstruct` at correlation 0.995 (0.999 smoothed, same amplitude) on EMPIAR-10499 series 00254.
Set `reconstruction.warp_compatible: false` for TeamTomo's own conventions.

## Validated on EMPIAR-10499 series 00254
* XML → tomograms vs real WarpTools output: 0.9948 / 0.999 (full resolution / smoothed), amplitude ratio 0.999.
* TSA → tomograms: gold-bead reprojection error 0.99 px (IMOD's fiducial alignment 0.73 px, Warp's AreTomo route
  2.47 px); tilt axis −4.50° from a blind start (IMOD −4.30°); defocus within 0.02 µm of Warp's `ts_ctf`.
* Limits: one tilt series tested; the hand (axis vs axis + 180°) can't be decided from the images
  (`alignment.flip_hand`); one defocus per tilt (no defocus gradient across tilted images).

## Install
Recommended: [uv](https://docs.astral.sh/uv/) — it installs `requirements.txt` in seconds.
```bash
git clone https://github.com/jtcarrion/teamtomo-workflows && cd teamtomo-workflows
pip install uv                                    # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # CUDA PyTorch (both from one index)
uv pip install -r requirements.txt
```

Without uv (same result, but slow: about 25 minutes, because pip clones the TeamTomo repository once for each of
the 18 pinned TeamTomo packages, where uv clones it once):
```bash
python -m venv .venv && source .venv/bin/activate            # Python >= 3.11
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```
Either way, pick the PyTorch build that matches your GPU driver (`cu126` here; see https://pytorch.org).

`requirements.txt` pins the TeamTomo packages to the commits these workflows were validated with;
`torch-tilt-series` comes from a fork until teamtomo/teamtomo#159 (`from_warp_xml`) is merged.
Tested from a clean environment: uv 4 s, pip 24 min for `requirements.txt`; both give the same packages, and both
workflows run on EMPIAR-10499 series 00254 (XML → tomograms vs WarpTools: correlation 0.995).

Contributing: `pip install pre-commit && pre-commit install` once — notebook outputs are then stripped on every commit.
