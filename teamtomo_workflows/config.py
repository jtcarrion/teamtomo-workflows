"""Run configuration: one YAML (or JSON) file per run, merged over these defaults.

Everything a run needs is in the file, so a person or an agent can reproduce it from the file alone; `run.json` in
every output folder stores the exact merged configuration that was used.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS: dict = {
    "workflow": None,                     # "xml_to_tomograms" or "tsa_to_tomograms"
    "input": {
        "warp_project": None,             # (a) Warp project folder (contains warp_tiltseries/, warp_frameseries/, tomostar/)
        "stacks": None,                   # (c) glob of tilt stacks, e.g. "/data/*.st", or a list of paths
        "tilt_angles": None,              # (c) per-stack angle file; default: <stack>.rawtlt, then <stack>.tlt
        "pixel_spacing": None,            # (c) stack pixel size in A; default: MRC header
        "series": "all",                  # "all" or a list of series names to run
    },
    "output": {
        "root": "teamtomo_output",        # everything goes in here, one sub-folder per series
        "overwrite": False,               # False: series with a finished run.json are skipped (resume)
    },
    "compute": {"device": "cuda"},
    "reconstruction": {
        "pixel_spacing": 10.0,            # output voxel size, A
        "thickness_angstrom": None,       # tomogram depth; default: Warp project's Tomo Z, else 1.5 x specimen slab
        "warp_compatible": True,          # Warp's conventions (matches WarpTools ts_reconstruct at corr 0.995)
        "ctf_and_dose": True,             # per-tilt CTF sign + dose weighting (needs defocus and dose; skipped if unknown)
        "images": "auto",                 # "auto": Warp averages if present, else the stack; or "averages" / "stack"
        "apply_local_shifts": True,       # xml_to_tomograms: apply Warp's 2-D GridMovement if the XML has one
    },
    "alignment": {                        # tsa_to_tomograms only
        "pixel_spacing": 10.0,            # coarse alignment at this spacing; projection matching at 2x
        "axis_prior": None,               # None: blind 0-180 deg search; or degrees; "tomostar": use Warp's prior
        "flip_hand": False,               # axis + 180 (mirror-image hand)
        "thickness_angstrom": None,       # specimen slab for projection matching; None: chosen from the data
        "thickness_candidates": [1500, 2000, 2500, 3000, 4000, 5000],
        "write_warp_xml": True,           # project inputs: also write the alignment as a Warp XML (back into WarpTools)
    },
    "ctf": {                              # tsa_to_tomograms only (xml_to_tomograms uses the XML's own CTF)
        "enabled": "auto",                # "auto": when full-size averages are available
        "voltage_kv": 300.0,
        "spherical_aberration_mm": 2.7,
        "amplitude_contrast": 0.07,
        "fit_range_angstrom": [30.0, 8.0],
        "defocus_range_um": [0.5, 6.0],
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k not in out:
            raise KeyError(f"unknown config key: {k!r} (known: {sorted(out)})")
        out[k] = _merge(out[k], v) if isinstance(out[k], dict) and isinstance(v, dict) else v
    return out


def load_config(path: Path | str | None = None, overrides: dict | None = None) -> dict:
    """Defaults <- file (YAML or JSON) <- overrides. Relative paths in the file are relative to the file."""
    data: dict = {}
    base_dir = Path.cwd()
    if path is not None:
        path = Path(path)
        base_dir = path.resolve().parent
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    cfg = _merge(DEFAULTS, data)
    cfg = _merge(cfg, overrides or {})
    for section, key in (("input", "warp_project"), ("output", "root")):
        v = cfg[section][key]
        if v is not None and not Path(v).is_absolute():
            cfg[section][key] = str((base_dir / v).resolve())
    if isinstance(cfg["input"]["stacks"], str) and not Path(cfg["input"]["stacks"]).is_absolute():
        cfg["input"]["stacks"] = str(base_dir / cfg["input"]["stacks"])
    if cfg["workflow"] not in ("xml_to_tomograms", "tsa_to_tomograms"):
        raise ValueError("config 'workflow' must be 'xml_to_tomograms' or 'tsa_to_tomograms'")
    if (cfg["input"]["warp_project"] is None) == (cfg["input"]["stacks"] is None):
        raise ValueError("set exactly one of input.warp_project (a Warp project) or input.stacks (plain tilt stacks)")
    if cfg["workflow"] == "xml_to_tomograms" and cfg["input"]["warp_project"] is None:
        raise ValueError("xml_to_tomograms needs input.warp_project (the XMLs live there)")
    return cfg
