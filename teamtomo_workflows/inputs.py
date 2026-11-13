"""Finding the inputs of every tilt series: (a) a Warp project, or (c) plain tilt stacks with angle files.

A `Series` holds paths and metadata only; nothing is loaded until a workflow asks for it.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import mrcfile
import numpy as np


@dataclass
class Series:
    name: str
    stack: Path | None                      # tilt stack (MRC), frames in tilt-series order
    tilt_angles: np.ndarray                 # nominal tilt angles (deg), stack order
    stack_pixel_spacing: float              # A; for Warp projects the TRUE spacing (from the XML's image size)
    xml: Path | None = None                 # Warp tilt-series XML (alignment, CTF, dose)
    averages: list[Path] = field(default_factory=list)  # full-size per-tilt averages, stack order (Warp projects)
    averages_pixel_spacing: float | None = None
    dose: np.ndarray | None = None          # cumulative dose per tilt (e/A^2), stack order
    axis_prior: float | None = None         # e.g. from Warp's tomostar
    image_dimensions_angstrom_xy: tuple[float, float] | None = None
    tomogram_dimensions_angstrom_xyz: tuple[float, float, float] | None = None  # Warp project's Tomo dimensions
    source: str = ""


# ---------------------------------------------------------------- (a) Warp project

def _settings(project: Path) -> dict:
    """Pixel size and Tomo dimensions from warp_tiltseries.settings (what ts_reconstruct uses)."""
    p = project / "warp_tiltseries.settings"
    if not p.exists():
        return {}
    s = p.read_text()
    get = lambda name: re.search(rf'Name="{name}" Value="([^"]+)"', s)
    out = {}
    if get("PixelSize"):
        out["pixel_size"] = float(get("PixelSize").group(1))
    dims = [get(f"Dimensions{a}") for a in "XYZ"]
    if all(dims) and "pixel_size" in out:
        out["tomo_dims_angstrom"] = tuple(float(d.group(1)) * out["pixel_size"] for d in dims)
    return out


def _tomostar(path: Path) -> dict:
    """Movie names, nominal tilt angles, prior axis angle and dose from a Warp .tomostar."""
    import starfile

    df = starfile.read(path)
    return {"movies": [Path(m).stem for m in df["wrpMovieName"]], "angles": np.array(df["wrpAngleTilt"], dtype=float),
            "axis": float(df["wrpAxisAngle"].iloc[0]) if "wrpAxisAngle" in df else None,
            "dose": np.array(df["wrpDose"], dtype=float) if "wrpDose" in df else None}


def warp_project_series(project: Path | str, names="all") -> list[Series]:
    import warpxmlfile

    project = Path(project)
    tsdir = project / "warp_tiltseries"
    settings = _settings(project)
    xmls = sorted(tsdir.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"no tilt-series XMLs in {tsdir}")
    out = []
    for xml in xmls:
        name = xml.stem
        if names != "all" and name not in [str(n) for n in names]:
            continue
        x = warpxmlfile.read(xml)
        block = xml.read_text()
        block = block[block.index("<MoviePath>") + 11:block.index("</MoviePath>")]
        movies = [Path(line.strip()).stem for line in block.splitlines() if line.strip()]
        averages = [project / "warp_frameseries" / "average" / f"{m}.mrc" for m in movies]
        averages = averages if all(a.exists() for a in averages) else []
        stack = tsdir / "tiltstack" / name / f"{name}.st"
        img_dims = tuple(float(v) for v in x.image_dimensions_angstrom)
        stack_px = None
        if stack.exists():
            with mrcfile.mmap(stack, permissive=True) as m:
                _, h, w = m.data.shape
            stack_px = float(np.mean([img_dims[0] / w, img_dims[1] / h]))     # true spacing, not the rounded header
        else:
            stack = None
        ts_star = project / "tomostar" / f"{name}.tomostar"
        star = _tomostar(ts_star) if ts_star.exists() else {}
        angles = star.get("angles", np.asarray(x.angles, float))
        dose = star.get("dose")
        dose = np.asarray(x.dose, float) if dose is None and getattr(x, "dose", None) is not None else dose
        out.append(Series(name=name, stack=stack, tilt_angles=np.array(angles, dtype=float), stack_pixel_spacing=stack_px or 0.0,
                          xml=xml, averages=averages, averages_pixel_spacing=settings.get("pixel_size"),
                          dose=dose, axis_prior=star.get("axis"), image_dimensions_angstrom_xy=img_dims,
                          tomogram_dimensions_angstrom_xyz=settings.get("tomo_dims_angstrom"), source=f"warp project {project}"))
    return out


# ---------------------------------------------------------------- (c) plain stacks

def plain_series(stacks, tilt_angles=None, pixel_spacing=None, names="all") -> list[Series]:
    paths = sorted(Path(p) for p in (glob.glob(stacks) if isinstance(stacks, str) else stacks))
    if not paths:
        raise FileNotFoundError(f"no stacks match {stacks}")
    out = []
    for st in paths:
        name = st.stem
        if names != "all" and name not in [str(n) for n in names]:
            continue
        if tilt_angles:
            angle_file = Path(str(tilt_angles).replace("{name}", name))
        else:
            angle_file = next((st.with_suffix(s) for s in (".rawtlt", ".tlt") if st.with_suffix(s).exists()), None)
        if angle_file is None or not angle_file.exists():
            raise FileNotFoundError(f"{st}: no tilt-angle file (looked for {name}.rawtlt / {name}.tlt; set input.tilt_angles)")
        with mrcfile.mmap(st, permissive=True) as m:
            n, h, w = m.data.shape
            px = float(pixel_spacing or m.voxel_size.x)
        if not px > 0:
            raise ValueError(f"{st}: no pixel size in the header; set input.pixel_spacing")
        angles = np.loadtxt(angle_file, dtype=float).reshape(-1)
        if len(angles) != n:
            raise ValueError(f"{st}: {n} frames but {len(angles)} angles in {angle_file}")
        out.append(Series(name=name, stack=st, tilt_angles=angles, stack_pixel_spacing=px,
                          image_dimensions_angstrom_xy=(w * px, h * px), source=f"plain stack {st}"))
    return out


def discover(cfg: dict) -> list[Series]:
    i = cfg["input"]
    if i["warp_project"]:
        return warp_project_series(i["warp_project"], i["series"])
    return plain_series(i["stacks"], i["tilt_angles"], i["pixel_spacing"], i["series"])
