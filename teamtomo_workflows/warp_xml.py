"""Writing a TeamTomo alignment back as a Warp tilt-series XML (so WarpTools / M can continue from it)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def alignment_to_warp_xml(ts, template_xml: Path, out_xml: Path, thickness_angstrom: float | None = None) -> Path:
    """Write ts (tilt angles, axis angle, sample_translations) into a copy of a Warp XML of the same series.

    Inverse of torch_tilt_series.from_warp_xml: AxisOffsetY/X = sample_translations (y, x) in Angstrom, AxisAngle =
    tilt_axis_angle, Angles = tilt_angles. Everything else (CTF, dose, movie paths) is kept from the template."""
    import warpxmlfile

    x = warpxmlfile.read(template_xml)
    t = ts.sample_translations.detach().cpu().numpy().astype(np.float64)
    x.axis_offset_y, x.axis_offset_x = t[:, 0].copy(), t[:, 1].copy()
    x.axis_angle = ts.tilt_axis_angle.detach().cpu().numpy().astype(np.float64)
    x.angles = ts.tilt_angles.detach().cpu().numpy().astype(np.float64)
    if thickness_angstrom is not None:
        d = x.volume_dimensions_angstrom                      # read-only property backed by the raw attribute
        x.attributes["VolumeDimensionsAngstrom"] = f"{d[0]}, {d[1]}, {float(thickness_angstrom)}"
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    x.to_file(out_xml)
    return out_xml
