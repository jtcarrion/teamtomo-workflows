"""Output layout and provenance. Everything lives under one root (default `teamtomo_output/`):

    teamtomo_output/
      batch_summary.csv               one row per series per batch run
      logs/                           batch logs
      <series>/
        <series>_<px>Apx.mrc          the tomogram
        run.json                      config used, package versions, GPU, timings, status, output files
        alignment.json                (tsa_to_tomograms) the alignment as a TeamTomo TiltSeries, Angstrom
        <series>.xml                  (tsa_to_tomograms, Warp projects) the alignment as a Warp XML -> WarpTools
        ctf.json                      (tsa_to_tomograms, when CTF ran) defocus per tilt
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import mrcfile
import numpy as np

PACKAGES = ("torch", "torch-tilt-series", "torch-reconstruct-tomogram", "torch-tiltxcorr", "torch-refine-tilt-axis-angle",
            "torch-fourier-slice", "torch-ctf-estimation", "torch-ctf", "warpxmlfile")


def series_dir(cfg: dict, name: str) -> Path:
    d = Path(cfg["output"]["root"]) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_done(cfg: dict, name: str) -> bool:
    p = Path(cfg["output"]["root"]) / name / "run.json"
    return p.exists() and json.loads(p.read_text()).get("status") == "done"


def versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


class Run:
    """Collects timings, results and output files for one series; written to run.json (also on failure)."""

    def __init__(self, cfg: dict, series):
        import torch

        self.cfg, self.series = cfg, series
        self.dir = series_dir(cfg, series.name)
        self.record = {"series": series.name, "workflow": cfg["workflow"], "status": "running",
                       "started": _dt.datetime.now().isoformat(timespec="seconds"), "source": series.source,
                       "config": cfg, "versions": versions(), "host": platform.node(),
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       "timings_s": {}, "results": {}, "outputs": {}}
        self._t0 = time.time()
        self.write()

    @contextmanager
    def step(self, name: str):
        t = time.time()
        yield
        self.record["timings_s"][name] = round(time.time() - t, 2)

    def result(self, **kv):
        self.record["results"].update({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in kv.items()})

    def output(self, key: str, path: Path):
        self.record["outputs"][key] = str(Path(path).relative_to(self.dir))

    def finish(self, status: str = "done", error: str | None = None):
        self.record.update(status=status, finished=_dt.datetime.now().isoformat(timespec="seconds"),
                           total_s=round(time.time() - self._t0, 1))
        if error:
            self.record["error"] = error
        self.write()

    def write(self):
        (self.dir / "run.json").write_text(json.dumps(self.record, indent=1, default=str))


def write_mrc(path: Path, volume: np.ndarray, pixel_spacing: float) -> Path:
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.asarray(volume, dtype=np.float32))
        m.voxel_size = pixel_spacing
    return path
