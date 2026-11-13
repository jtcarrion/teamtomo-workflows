"""The two workflows (one tilt series each) and the batch loop over many series.

    xml_to_tomograms : Warp/M/MissAlignment XML + images  -> tomogram (Warp-compatible TeamTomo reconstruction)
    tsa_to_tomograms : images + tilt angles only           -> TeamTomo alignment (+ CTF) -> tomogram (+ Warp XML)
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import traceback
from pathlib import Path

import mrcfile
import numpy as np
import torch

from . import outputs as O
from . import tsa as T
from . import warp_compat as WC
from .inputs import Series, discover

log = logging.getLogger("teamtomo_workflows")


# ---------------------------------------------------------------- images

def load_images(s: Series, target_px: float, mode: str = "auto", device: str = "cuda") -> tuple[torch.Tensor, float]:
    """Tilt images at ~target_px, in stack order, and their TRUE pixel spacing.

    "averages" (Warp projects): each full-size average Fourier-cropped to Warp's own size, round(d / factor / 2) * 2,
    as ts_reconstruct does; "stack": the tilt stack, Fourier-binned if coarser than its own spacing.
    The spacing returned is the physical image size / pixel count (not a rounded header value)."""
    from torch_fourier_rescale import fourier_rescale_2d

    use_avg = mode == "averages" or (mode == "auto" and s.averages)
    if use_avg:
        if not s.averages:
            raise FileNotFoundError(f"{s.name}: no full-size averages found")
        imgs = []
        for p in s.averages:
            with mrcfile.mmap(p, permissive=True) as m:
                im = torch.tensor(np.asarray(m.data, dtype=np.float32), device=device)
            f = target_px / s.averages_pixel_spacing
            shape = (int(round(im.shape[0] / f / 2)) * 2, int(round(im.shape[1] / f / 2)) * 2)
            out, _ = fourier_rescale_2d(im, target_shape=shape)
            imgs.append(out)
        images = torch.stack(imgs)
    else:
        if s.stack is None:
            raise FileNotFoundError(f"{s.name}: no tilt stack found")
        images, px = T.read_stack(s.stack, s.stack_pixel_spacing or None, device=device)
        images, _ = T.bin_to(images, px, target_px)
    h, w = images.shape[-2:]
    if s.image_dimensions_angstrom_xy is not None:
        dx, dy = s.image_dimensions_angstrom_xy
        px_true = float(np.mean([dx / w, dy / h]))
    else:
        px_true = float(target_px)
    return images, px_true


def _volume_dims(cfg: dict, s: Series, image_dims_xy, fallback_z: float | None) -> tuple[float, float, float]:
    z = cfg["reconstruction"]["thickness_angstrom"]
    if z is None and s.tomogram_dimensions_angstrom_xyz is not None:
        return tuple(s.tomogram_dimensions_angstrom_xyz)
    if z is None:
        z = fallback_z
    if z is None:
        raise ValueError(f"{s.name}: no tomogram thickness; set reconstruction.thickness_angstrom")
    return (float(image_dims_xy[0]), float(image_dims_xy[1]), float(z))


def _reconstruct(cfg, s, ts, images, px_img, vol_dims, dose, defocus):
    rc = cfg["reconstruction"]
    voxel = float(rc["pixel_spacing"])
    image_dims = s.image_dimensions_angstrom_xy or (images.shape[-1] * px_img, images.shape[-2] * px_img)
    if rc["warp_compatible"]:
        ctf = bool(rc["ctf_and_dose"]) and dose is not None and defocus is not None
        vol = WC.reconstruct_warp_compatible(ts, images, vol_dims, image_dims, voxel_size=voxel, dose=dose,
                                             defocus_um=defocus, ctf_and_dose=ctf)
        return vol.cpu().numpy(), {"warp_compatible": True, "ctf_and_dose_applied": ctf}
    shape = tuple(int(round(d / voxel / 2)) * 2 for d in vol_dims)[::-1]
    t = ts
    t.pixel_spacing = px_img
    return T.reconstruct(t, images, shape), {"warp_compatible": False, "ctf_and_dose_applied": False}


# ---------------------------------------------------------------- workflow 1: XML -> tomograms

def xml_to_tomograms(cfg: dict, s: Series, device: str = "cuda") -> dict:
    from torch_tilt_series import from_warp_xml

    run = O.Run(cfg, s)
    try:
        rc = cfg["reconstruction"]
        with run.step("load_images"):
            images, px = load_images(s, rc["pixel_spacing"], rc["images"], device)
        with run.step("read_xml"):
            ts = from_warp_xml(s.xml, image_path=s.stack, pixel_spacing=px, device=device,
                               apply_local_shifts=rc["apply_local_shifts"])
            P = WC.warp_xml_parameters(s.xml)
            xml_z = P["volume_dimensions_angstrom_xyz"][2]
            vol_dims = _volume_dims(cfg, s, P["image_dimensions_angstrom_xy"], xml_z if xml_z > 10 else None)
        with run.step("reconstruct"):
            vol, how = _reconstruct(cfg, s, ts, images, px, vol_dims, P["dose"], P["defocus_um"])
        out = O.write_mrc(run.dir / f"{s.name}_{rc['pixel_spacing']:.2f}Apx.mrc", vol, rc["pixel_spacing"])
        run.output("tomogram", out)
        run.result(volume_shape_zyx=list(vol.shape), volume_dimensions_angstrom_xyz=list(vol_dims),
                   image_pixel_spacing=px, local_grid=ts.local_shifts_2d is not None, **how)
        run.finish()
    except Exception as e:
        run.finish("failed", f"{e}\n{traceback.format_exc()}")
        raise
    return run.record


# ---------------------------------------------------------------- workflow 2: tilt-series alignment -> tomograms

def tsa_to_tomograms(cfg: dict, s: Series, device: str = "cuda") -> dict:
    run = O.Run(cfg, s)
    try:
        ac, rc, cc = cfg["alignment"], cfg["reconstruction"], cfg["ctf"]
        with run.step("load_alignment_images"):
            img_al, px_al = load_images(s, ac["pixel_spacing"], "stack" if s.stack else "averages", device)
        prior = s.axis_prior if ac["axis_prior"] == "tomostar" else ac["axis_prior"]
        with run.step("coarse_alignment"):
            al = T.align(img_al, s.tilt_angles, px_al, axis_prior=prior, flip_hand=ac["flip_hand"])
            ts = T.to_tilt_series(al, image_path=s.stack, image_pixel_spacing=px_al)
        run.result(tilt_axis_angle_deg=al.tilt_axis_angle, coarse_rounds=len(al.history) - 1)
        with run.step("projection_matching"):
            if ac["thickness_angstrom"] is None:
                thickness, pm, table = T.choose_thickness(ts, img_al, candidates_angstrom=list(ac["thickness_candidates"]))
                run.result(thickness_scores=[[float(t), float(v)] for t, v in table])
            else:
                thickness = float(ac["thickness_angstrom"])
                pm = T.refine_by_projection_matching(ts, img_al, thickness, verbose=False)
            ts = pm.tilt_series
        run.result(specimen_thickness_angstrom=thickness, projection_matching=pm.history)
        if rc["warp_compatible"]:                                   # Warp's tomogram orientation (as from_warp_xml)
            ts.levelled2tomo = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], device=ts.sample_translations.device))
        T.save_alignment(ts, run.dir / "alignment.json", specimen_thickness_angstrom=thickness,
                         alignment_pixel_spacing_angstrom=px_al, history=al.history + pm.history)
        run.output("alignment", run.dir / "alignment.json")

        defocus = None
        if cc["enabled"] is True or (cc["enabled"] == "auto" and s.averages):
            with run.step("ctf"):
                rows = T.estimate_ctf_per_tilt(s.averages, s.averages_pixel_spacing, cc["voltage_kv"], cc["spherical_aberration_mm"],
                                               cc["amplitude_contrast"], tuple(cc["fit_range_angstrom"]), tuple(cc["defocus_range_um"]),
                                               verbose=False)
            (run.dir / "ctf.json").write_text(json.dumps(rows, indent=1, default=float))
            run.output("ctf", run.dir / "ctf.json")
            defocus = np.array([r["defocus_um"] for r in rows])
            run.result(mean_defocus_um=float(np.mean(defocus)), ctf_fits_flagged=int(sum(not r["reliable"] for r in rows)))

        with run.step("load_reconstruction_images"):
            images, px = load_images(s, rc["pixel_spacing"], rc["images"], device)
        image_dims = s.image_dimensions_angstrom_xy or (images.shape[-1] * px, images.shape[-2] * px)
        vol_dims = _volume_dims(cfg, s, image_dims, 1.5 * thickness)
        with run.step("reconstruct"):
            vol, how = _reconstruct(cfg, s, ts, images, px, vol_dims, s.dose, defocus)
        out = O.write_mrc(run.dir / f"{s.name}_{rc['pixel_spacing']:.2f}Apx.mrc", vol, rc["pixel_spacing"])
        run.output("tomogram", out)
        run.result(volume_shape_zyx=list(vol.shape), volume_dimensions_angstrom_xyz=list(vol_dims), **how)

        if s.xml is not None and ac["write_warp_xml"]:
            from .warp_xml import alignment_to_warp_xml
            xml_out = alignment_to_warp_xml(ts, s.xml, run.dir / f"{s.name}.xml", thickness_angstrom=vol_dims[2])
            run.output("warp_xml", xml_out)
        run.finish()
    except Exception as e:
        run.finish("failed", f"{e}\n{traceback.format_exc()}")
        raise
    return run.record


WORKFLOWS = {"xml_to_tomograms": xml_to_tomograms, "tsa_to_tomograms": tsa_to_tomograms}


# ---------------------------------------------------------------- batch

def run_batch(cfg: dict, stop_on_error: bool = False) -> list[dict]:
    """Every series in the config, one after another on one GPU. Finished series are skipped unless
    output.overwrite; a failing series is recorded (run.json, batch_summary.csv, log) and the batch continues."""
    root = Path(cfg["output"]["root"]); (root / "logs").mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    handler = logging.FileHandler(root / "logs" / f"batch_{stamp}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler); log.setLevel(logging.INFO)
    fn = WORKFLOWS[cfg["workflow"]]
    series = discover(cfg)
    log.info("%s: %d series from %s", cfg["workflow"], len(series), series[0].source if series else "-")
    rows = []
    for i, s in enumerate(series, 1):
        if not cfg["output"]["overwrite"] and O.is_done(cfg, s.name):
            log.info("[%d/%d] %s: done already, skipped", i, len(series), s.name)
            rows.append({"series": s.name, "status": "skipped (done)"})
            continue
        log.info("[%d/%d] %s: start", i, len(series), s.name)
        try:
            rec = fn(cfg, s, cfg["compute"]["device"])
            rows.append({"series": s.name, "status": rec["status"], "total_s": rec.get("total_s"),
                         **{k: rec["results"].get(k) for k in ("tilt_axis_angle_deg", "specimen_thickness_angstrom")}})
            log.info("[%d/%d] %s: done in %ss", i, len(series), s.name, rec.get("total_s"))
        except Exception as e:  # keep going; details are in <series>/run.json
            rows.append({"series": s.name, "status": "failed", "error": str(e).splitlines()[0][:200]})
            log.error("[%d/%d] %s: FAILED: %s", i, len(series), s.name, e)
            if stop_on_error:
                raise
        finally:
            torch.cuda.empty_cache()
    summary = root / "batch_summary.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "series", k != "status", k))
    new = not summary.exists()
    with summary.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch"] + keys, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"batch": stamp, **r})
    log.removeHandler(handler)
    return rows
