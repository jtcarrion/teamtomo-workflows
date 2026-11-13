"""Tilt-series alignment (TSA) and CTF for the TeamTomo workflows.

TeamTomo pieces (all from the monorepo, nothing else does any alignment):
    torch_tiltxcorr.tiltxcorr                           per-tilt shifts (stretched cross-correlation of adjacent tilts)
    torch_refine_tilt_axis_angle.refine_tilt_axis_angle tilt-axis angle (common-line coherence)
    torch_tilt_series.TiltSeries                        the geometry the reconstruction consumes
    torch_reconstruct_tomogram.reconstruct_tomogram     the tomogram

Everything here is glue: units, conventions, the shift <-> axis alternation. Conventions were established on
EMPIAR-10499 00254 against IMOD's fiducial solution (see README.md):
    * TiltSeries.sample_translations (Angstrom, yx) = -tiltxcorr shifts (px, yx) * pixel spacing (+ any constant)
    * refine_tilt_axis_angle uses TiltSeries' sense of the axis angle, modulo 180 deg
    * axis a and a + 180 fit the data equally well but mean opposite tilt-angle signs (mirror-image tomograms);
      we keep the representative in (-90, 90] (IMOD/Warp's choice for 00254) unless told otherwise (`flip_hand`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mrcfile
import numpy as np
import torch
from torch_fourier_rescale import fourier_rescale_2d
from torch_refine_tilt_axis_angle import refine_tilt_axis_angle
from torch_tilt_series import TiltSeries
from torch_tiltxcorr import tiltxcorr

# ---------------------------------------------------------------------------------------------- input


def read_stack(path: Path | str, pixel_spacing: float | None = None, device: str = "cpu") -> tuple[torch.Tensor, float]:
    """(n, h, w) float32 tensor and its pixel spacing in Angstrom (from the MRC header unless given)."""
    with mrcfile.mmap(path, permissive=True) as m:
        data = torch.tensor(np.asarray(m.data, dtype=np.float32), device=device)
        header_px = float(m.voxel_size.x)
    px = pixel_spacing or header_px
    if not px > 0:
        raise ValueError(f"{path}: no pixel size in the MRC header, pass pixel_spacing")
    return data, px


def read_tilt_angles(path: Path | str) -> np.ndarray:
    """Tilt angles in degrees, one per frame, from an IMOD .rawtlt / .tlt (one number per line)."""
    return np.loadtxt(path, dtype=np.float64).reshape(-1)


def bin_to(stack: torch.Tensor, pixel_spacing: float, target_pixel_spacing: float) -> tuple[torch.Tensor, float]:
    """Fourier-crop the stack to ~target_pixel_spacing (no-op if it is already coarser). Returns (stack, new spacing)."""
    if target_pixel_spacing <= pixel_spacing * 1.01:
        return stack, pixel_spacing
    binned, (new_px, _) = fourier_rescale_2d(stack, source_spacing=pixel_spacing, target_spacing=target_pixel_spacing)
    return binned, float(new_px)


# ------------------------------------------------------------------------------------------ alignment


def canonical_axis(axis_deg: float, flip_hand: bool = False) -> float:
    """Representative of axis_deg (mod 180) in (-90, 90]; flip_hand adds 180 (the mirror-image hand)."""
    a = (axis_deg + 90.0) % 180.0 - 90.0
    a = 90.0 if a == -90.0 else a
    return a + 180.0 if flip_hand else a


@dataclass
class Alignment:
    """Result of `align`: everything in the pixel spacing the alignment ran at, plus its history."""

    tilt_angles: np.ndarray                    # (n,) degrees
    tilt_axis_angle: float                     # degrees, TiltSeries convention
    shifts_px: np.ndarray                      # (n, 2) yx, tiltxcorr convention (sample_translations = -shifts * px)
    pixel_spacing: float                       # Angstrom, of the stack the alignment ran on
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def align(
    stack: torch.Tensor,
    tilt_angles: np.ndarray,
    pixel_spacing: float,
    axis_prior: float | None = None,
    n_iterations: int = 4,
    tolerance_deg: float = 0.05,
    lowpass_angstroms: float | None = None,
    flip_hand: bool = False,
) -> Alignment:
    """Alternate tilt-axis refinement and tiltxcorr until the axis stops moving.

    The two algorithms need each other: tiltxcorr stretches perpendicular to the axis, and the common-line axis
    search wants a translationally aligned stack. Start: axis from the *unaligned* stack (blind 0-180 deg search,
    or +-90 deg around `axis_prior`), then shifts, then the axis on the shifted stack, and so on.
    """
    from torch_fourier_shift import fourier_shift_image_2d

    t0 = time.time()
    angles = torch.as_tensor(tilt_angles, dtype=torch.float32, device=stack.device)
    prior = 90.0 if axis_prior is None else float(axis_prior)
    axis = refine_tilt_axis_angle(stack, tilt_axis_angle=prior)
    history = [{"iteration": 0, "axis_raw": axis, "on": "unaligned stack"}]
    shifts = torch.zeros(len(angles), 2, device=stack.device)
    for it in range(1, n_iterations + 1):
        shifts = tiltxcorr(stack, angles, tilt_axis_angle=axis, pixel_spacing_angstroms=pixel_spacing,
                           lowpass_angstroms=lowpass_angstroms)
        aligned = fourier_shift_image_2d(stack, shifts=shifts)
        new_axis = refine_tilt_axis_angle(aligned, tilt_axis_angle=axis)
        history.append({"iteration": it, "axis_raw": new_axis, "on": "tiltxcorr-aligned stack",
                        "max_shift_px": float(shifts.norm(dim=1).max())})
        converged = abs(new_axis - axis) < tolerance_deg
        axis = new_axis
        if converged:
            break
    # final shifts at the final axis (cheap, and keeps shifts and axis consistent)
    shifts = tiltxcorr(stack, angles, tilt_axis_angle=axis, pixel_spacing_angstroms=pixel_spacing,
                       lowpass_angstroms=lowpass_angstroms)
    return Alignment(
        tilt_angles=np.asarray(tilt_angles, dtype=np.float64),
        tilt_axis_angle=canonical_axis(axis, flip_hand),
        shifts_px=shifts.cpu().numpy(),
        pixel_spacing=pixel_spacing,
        history=history,
        seconds=time.time() - t0,
    )


def to_tilt_series(
    alignment: Alignment,
    image_path: Path | str | None = None,
    image_pixel_spacing: float | None = None,
    device: str = "cpu",
) -> TiltSeries:
    """The alignment as a TiltSeries in Angstrom, for images at `image_path` (any binning of the aligned stack).

    sample_translations = -shifts * pixel spacing: tiltxcorr reports how far each image must move to line up with
    the 0-degree image; TiltSeries wants where the sample origin lands on the detector. The constant this leaves
    free (tiltxcorr pins the 0-degree tilt) only moves the tomogram's origin; we keep the 0-degree image centre.
    """
    n = len(alignment.tilt_angles)
    translations = -torch.as_tensor(alignment.shifts_px, dtype=torch.float32) * alignment.pixel_spacing
    return TiltSeries(
        tilt_angles=torch.as_tensor(alignment.tilt_angles, dtype=torch.float32),
        tilt_axis_angle=torch.full((n,), alignment.tilt_axis_angle, dtype=torch.float32),
        sample_translations=translations,
        image_path=image_path,
        pixel_spacing=image_pixel_spacing or alignment.pixel_spacing,
        device=device,
    )


# --------------------------------------------------------------------------------------- reconstruction


def reconstruct(ts: TiltSeries, images: torch.Tensor, volume_shape: tuple[int, int, int], output: Path | None = None,
                sidelength: int = 128, batch_size: int = 4, **preprocessing) -> np.ndarray:
    """reconstruct_tomogram from in-memory images; optional MRC output at the images' pixel spacing."""
    from torch_reconstruct_tomogram import reconstruct_tomogram

    if str(ts.sample_translations.device) != str(images.device):
        ts.to(images.device)
    vol = reconstruct_tomogram(ts, volume_shape=volume_shape, sidelength=sidelength, batch_size=batch_size,
                               images=images, **preprocessing).cpu().numpy()
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with mrcfile.new(output, overwrite=True) as m:
            m.set_data(vol.astype(np.float32))
            m.voxel_size = ts.pixel_spacing
    return vol


# ------------------------------------------------------------- validation (optional: needs a reference)


def gauge_free_shift_error(ts: TiltSeries, reference: TiltSeries) -> tuple[np.ndarray, np.ndarray]:
    """Per-tilt |translation difference| (px) after removing the best 3-D sample-origin offset.

    Two alignments are only comparable up to where each puts the sample origin; an origin (ox, oy, oz) shows up
    on the detector as Rz(axis) [oy, ox cos(theta) + oz sin(theta)]. Returns (per-tilt error px, offset A xyz)."""
    th = np.deg2rad(reference.tilt_angles.cpu().numpy())
    a = np.deg2rad(reference.tilt_axis_angle.cpu().numpy())
    rows = []
    for t, ai in zip(th, a):
        ex, ey = np.array([np.cos(t), 0, np.sin(t)]), np.array([0, 1, 0])
        rows.append(np.stack([np.sin(ai) * ex + np.cos(ai) * ey, np.cos(ai) * ex - np.sin(ai) * ey]))
    A = np.concatenate(rows)
    d = (ts.sample_translations.cpu() - reference.sample_translations.cpu()).numpy().reshape(-1)
    p, *_ = np.linalg.lstsq(A, d, rcond=None)
    r = (d - A @ p).reshape(-1, 2) / ts.pixel_spacing
    return np.linalg.norm(r, axis=1), p


def bead_observations(tiltstack_dir: Path, name: str):
    """IMOD bead tracks in raw-stack pixels: (contour, view, xy_px, centre). The .fid was tracked on the
    prealigned stack, so .prexg is undone (same as phase 1)."""
    import imodmodel

    fid = imodmodel.read(Path(tiltstack_dir) / f"{name}.fid")
    with mrcfile.mmap(Path(tiltstack_dir) / f"{name}.st", permissive=True) as m:
        _, ny, nx = m.data.shape
    centre = np.array([nx / 2.0, ny / 2.0])
    xf = np.loadtxt(Path(tiltstack_dir) / f"{name}.prexg").reshape(-1, 6)
    view = fid["z"].round().astype(int).to_numpy()
    a, d = xf[view][:, :4].reshape(-1, 2, 2), xf[view][:, 4:6]
    xy = np.einsum("nij,nj->ni", np.linalg.inv(a), fid[["x", "y"]].to_numpy() - centre - d) + centre
    return fid["contour_id"].to_numpy(), view, xy, centre


def bead_residuals(ts: TiltSeries, contour, view, xy_px, centre) -> np.ndarray:
    """Triangulate every bead through the TiltSeries (least squares) and return per-observation residuals (px).
    Independent of the alignment's choice of origin, so any two alignments are directly comparable."""
    basis = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32) * 1000.0
    with torch.no_grad():
        q = ts.project_points(basis.flip(-1).to(ts.device)).cpu().numpy()[..., ::-1]   # xyz in -> xy out (A)
    b = q[0]
    A = np.stack([(q[k] - b) / 1000.0 for k in (1, 2, 3)], -1)                          # (T, 2, 3)
    obs = (xy_px - centre) * ts.pixel_spacing
    resid = np.zeros_like(obs)
    for bead in np.unique(contour):
        m = contour == bead
        p, *_ = np.linalg.lstsq(A[view[m]].reshape(-1, 3), (obs[m] - b[view[m]]).reshape(-1), rcond=None)
        resid[m] = obs[m] - (A[view[m]] @ p + b[view[m]])
    return np.linalg.norm(resid, axis=1) / ts.pixel_spacing


def match_origin(ts: TiltSeries, reference: TiltSeries) -> TiltSeries:
    """Copy of ts whose sample origin is moved onto the reference's (the offset `gauge_free_shift_error` fits).
    Only for validation: lets two tomograms from different alignments be overlaid voxel for voxel."""
    import copy

    _, p = gauge_free_shift_error(ts, reference)
    th = np.deg2rad(reference.tilt_angles.cpu().numpy())
    a = np.deg2rad(reference.tilt_axis_angle.cpu().numpy())
    sx = p[0] * np.cos(th) + p[2] * np.sin(th)            # scope x of the offset
    sy = np.full_like(th, p[1])
    dy, dx = np.sin(a) * sx + np.cos(a) * sy, np.cos(a) * sx - np.sin(a) * sy
    out = copy.copy(ts)
    out.sample_translations = ts.sample_translations - torch.tensor(np.stack([dy, dx], 1), dtype=torch.float32,
                                                                     device=ts.sample_translations.device)
    out.levelled2tomo = reference.levelled2tomo.clone().to(ts.sample_translations.device)
    return out


# ------------------------------------------------------------------------------- projection matching


def _gauge_basis(ts: TiltSeries) -> np.ndarray:
    """(2n, 3) detector-yx response (per unit Angstrom) of moving the sample origin by (ox, oy, oz)."""
    th = np.deg2rad(ts.tilt_angles.cpu().numpy())
    a = np.deg2rad(ts.tilt_axis_angle.cpu().numpy())
    rows = []
    for t, ai in zip(th, a):
        ex, ey = np.array([np.cos(t), 0, np.sin(t)]), np.array([0, 1, 0])
        rows.append(np.stack([np.sin(ai) * ex + np.cos(ai) * ey, np.cos(ai) * ex - np.sin(ai) * ey]))
    return np.concatenate(rows)


def _cosine_window(h: int, w: int, edge: float, device) -> torch.Tensor:
    """1 in the middle, cosine fall-off over `edge` (fraction of each side) to 0 at the border."""
    def ramp(n):
        x = torch.linspace(0, 1, n, device=device)
        e = max(edge, 1e-3)
        r = torch.clamp(torch.minimum(x, 1 - x) / e, 0, 1)
        return 0.5 - 0.5 * torch.cos(torch.pi * r)
    return torch.outer(ramp(h), ramp(w))


@dataclass
class ProjectionMatching:
    tilt_series: TiltSeries                    # refined copy (translations updated)
    history: list[dict]                        # per pass: mode, rms / max correction (px at the input spacing)
    corrections_px: list[np.ndarray]           # per pass (n, 2) yx, input pixels, origin move removed
    seconds: float


class _Matcher:
    """Shared machinery: binned + preprocessed images, cube side S, specimen slab, reprojection, cross-correlation."""

    def __init__(self, ts: TiltSeries, images: torch.Tensor, thickness_angstrom: float, binning: int,
                 lowpass_angstrom: float | None, max_correction_px: float):
        import torch.nn.functional as F
        from torch_tilt_series import preprocess_tilt_series_images

        self.F = F
        self.ts, self.device = ts, images.device
        self.px_in = ts.pixel_spacing
        work, self.px = bin_to(images, self.px_in, self.px_in * binning) if binning > 1 else (images, self.px_in)
        high = 0.5 if lowpass_angstrom is None else min(0.5, self.px / lowpass_angstrom)
        self.work = preprocess_tilt_series_images(work, low=0.025, high=high, falloff=0.025)
        n, h, w = self.work.shape
        self.S = S = int(np.ceil(max(h, w) / 16) * 16)
        self.pads = [(S - w) // 2, S - w - (S - w) // 2, (S - h) // 2, S - h - (S - h) // 2]
        self.centre_offset = torch.tensor([S // 2 - (h // 2 + self.pads[2]), S // 2 - (w // 2 + self.pads[0])],
                                          dtype=torch.float32, device=self.device)
        self.window = F.pad(_cosine_window(h, w, 0.1, self.device), self.pads)
        zz = torch.arange(S, device=self.device, dtype=torch.float32) - S // 2
        half = 0.5 * thickness_angstrom / self.px
        self.slab = torch.clamp((half + 4 - zz.abs()) / 8, 0, 1)[:, None, None]
        self.r = int(min(max(2, max_correction_px * self.px_in / self.px), S // 4))
        self.scores: list[float] = []

    def geometry(self):
        """Fourier-insertion rotations (as torch_reconstruct_tomogram) and the projected origin per tilt (px)."""
        ts = self.ts
        rot = torch.linalg.pinv(ts.projection_matrices[:, :3, :3] @ ts.tomo2sample[:3, :3]).float()
        origin = ts.project_points(torch.zeros(1, 3, device=self.device))[0] / self.px
        return rot, origin

    def centred(self, origin: torch.Tensor) -> torch.Tensor:
        """Images (tapered, padded to S) shifted so the projected sample origin sits at (S//2, S//2)."""
        from torch_fourier_shift import fourier_shift_image_2d

        img = fourier_shift_image_2d(self.F.pad(self.work, self.pads) * self.window, shifts=self.centre_offset - origin)
        return img - img.mean(dim=(-2, -1), keepdim=True)

    @staticmethod
    def rfft(img: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(img, dim=(-2, -1))), dim=(-2,))

    def insert(self, rfft: torch.Tensor, rot: torch.Tensor):
        from torch_fourier_slice import insert_central_slices_rfft_3d

        return insert_central_slices_rfft_3d(rfft, volume_shape=(self.S,) * 3, rotation_matrices=rot,
                                             zyx_matrices=True, fftfreq_max=0.5)

    def reproject(self, vol: torch.Tensor, wts: torch.Tensor, rot_i: torch.Tensor) -> torch.Tensor:
        """Normalise the Fourier volume, keep only the specimen slab in real space, project at one tilt."""
        from torch_fourier_slice import extract_central_slices_rfft_3d

        S = self.S
        ref = vol / torch.clamp(wts, min=1.0)
        real = torch.fft.fftshift(torch.fft.irfftn(torch.fft.ifftshift(ref, dim=(-3, -2)), s=(S,) * 3), dim=(-3, -2, -1))
        ref = torch.fft.fftshift(torch.fft.rfftn(torch.fft.ifftshift(real * self.slab, dim=(-3, -2, -1))), dim=(-3, -2))
        p = extract_central_slices_rfft_3d(ref, rotation_matrices=rot_i[None], zyx_matrices=True, fftfreq_max=0.5)
        return torch.fft.fftshift(torch.fft.irfft2(torch.fft.ifftshift(p, dim=(-2,)), s=(S, S)), dim=(-2, -1))[0]

    def measure(self, img_i: torch.Tensor, proj: torch.Tensor, origin_i: torch.Tensor) -> torch.Tensor:
        """Where image i sits relative to its reprojection: (dy, dx) in working pixels (sub-pixel)."""
        from torch_fourier_shift import fourier_shift_image_2d

        S, r = self.S, self.r
        m = fourier_shift_image_2d(self.window, shifts=self.centre_offset - origin_i)
        a, b = img_i * m, proj * m
        a, b = (a - a.mean()) / (a.std() + 1e-6), (b - b.mean()) / (b.std() + 1e-6)
        cc = torch.fft.fftshift(torch.fft.irfft2(torch.fft.rfft2(a) * torch.conj(torch.fft.rfft2(b)), s=(S, S)))
        c = S // 2
        win = cc[c - r:c + r + 1, c - r:c + r + 1]
        ky, kx = divmod(int(torch.argmax(win)), win.shape[1])
        self.scores.append(float(win[ky, kx]) / float((m > 0.5).sum()))     # ~ correlation coefficient

        def sub(vm, v0, vp):
            d = float(vm - 2 * v0 + vp)
            return 0.0 if abs(d) < 1e-12 else float(0.5 * (vm - vp)) / d
        dy = ky - r + (sub(win[ky - 1, kx], win[ky, kx], win[ky + 1, kx]) if 0 < ky < 2 * r else 0.0)
        dx = kx - r + (sub(win[ky, kx - 1], win[ky, kx], win[ky, kx + 1]) if 0 < kx < 2 * r else 0.0)
        return torch.tensor([dy, dx], dtype=torch.float32, device=self.device)

    def pass_outward(self, n_seed: int = 3) -> torch.Tensor:
        """Increasing |tilt|: each tilt is matched to a reprojection of the lower tilts only (already corrected),
        then inserted. Correlated errors of neighbouring high tilts cannot vouch for each other this way."""
        from torch_fourier_shift import fourier_shift_image_2d

        rot, origin = self.geometry()
        img = self.centred(origin)
        order = torch.argsort(self.ts.tilt_angles.abs()).tolist()
        corr = torch.zeros(len(order), 2, device=self.device)
        vol, wts = self.insert(self.rfft(img[order[:n_seed]]), rot[order[:n_seed]])
        for i in order[n_seed:]:
            d = self.measure(img[i], self.reproject(vol, wts, rot[i]), origin[i])
            corr[i] = d
            vi, wi = self.insert(self.rfft(fourier_shift_image_2d(img[i], shifts=-d))[None], rot[i:i + 1])
            vol, wts = vol + vi, wts + wi
        return corr

    def pass_leave_one_out(self) -> torch.Tensor:
        """Every tilt against a reprojection of all the others (polish once the gross errors are gone)."""
        rot, origin = self.geometry()
        img = self.centred(origin)
        rfft = self.rfft(img)
        vol, wts = self.insert(rfft, rot)
        corr = torch.zeros(len(img), 2, device=self.device)
        for i in range(len(img)):
            vi, wi = self.insert(rfft[i:i + 1], rot[i:i + 1])
            corr[i] = self.measure(img[i], self.reproject(vol - vi, wts - wi, rot[i]), origin[i])
        return corr


def refine_by_projection_matching(
    ts: TiltSeries,
    images: torch.Tensor,
    thickness_angstrom: float,
    n_outward: int = 3,
    n_leave_one_out: int = 3,
    binning: int = 2,
    lowpass_angstrom: float | None = None,
    max_correction_px: float = 20.0,
    verbose: bool = True,
) -> ProjectionMatching:
    """Refine per-tilt translations by matching each tilt to a reprojection of other tilts.

    Built only from TeamTomo pieces: the TiltSeries geometry (the same rotations torch_reconstruct_tomogram uses
    for Fourier insertion), torch_fourier_slice central-slice insertion / extraction, torch_fourier_shift.

    1. Centre every image on its projected sample origin, taper, pad to a cube side S (at `binning` x the
       input pixel spacing).
    2. `n_outward` passes in increasing |tilt|: the reference for tilt i is the reconstruction from the lower
       tilts only, restricted to the specimen slab |z| < thickness/2, reprojected at tilt i (AreTomo's idea).
    3. `n_leave_one_out` passes: reference = all other tilts (polish).
    4. After each pass, remove the part of the corrections that only moves the sample origin (a free choice),
       and update `sample_translations`.
    """
    import copy

    t0 = time.time()
    out = copy.copy(ts)
    out.to(images.device)
    mt = _Matcher(out, images, thickness_angstrom, binning, lowpass_angstrom, max_correction_px)
    G = _gauge_basis(ts)
    history, corrections = [], []
    for mode in ["outward"] * n_outward + ["leave_one_out"] * n_leave_one_out:
        mt.scores = []
        corr = mt.pass_outward() if mode == "outward" else mt.pass_leave_one_out()
        d = (corr * mt.px).cpu().numpy().reshape(-1)
        p, *_ = np.linalg.lstsq(G, d, rcond=None)
        d = (d - G @ p).reshape(-1, 2)
        out.sample_translations = out.sample_translations + torch.tensor(d, dtype=torch.float32, device=images.device)
        c_px = np.linalg.norm(d, axis=1) / mt.px_in
        corrections.append(d / mt.px_in)
        history.append({"pass": len(history) + 1, "mode": mode, "rms_correction_px": float(np.sqrt((c_px ** 2).mean())),
                        "max_correction_px": float(c_px.max()), "mean_cc": float(np.mean(mt.scores)),
                        "seconds": time.time() - t0})
        if verbose:
            h = history[-1]
            print(f"pass {h['pass']} ({mode}): rms correction {h['rms_correction_px']:.2f} px, "
                  f"max {h['max_correction_px']:.2f} px ({h['seconds']:.0f} s)")
    return ProjectionMatching(tilt_series=out, history=history, corrections_px=corrections, seconds=time.time() - t0)


def choose_thickness(
    ts: TiltSeries,
    images: torch.Tensor,
    candidates_angstrom: list[float] | None = None,
    refine_step_angstrom: float | None = 200.0,
    **pm_kwargs,
) -> tuple[float, ProjectionMatching, list[tuple[float, float]]]:
    """Pick the specimen-slab thickness for projection matching from the data alone.

    Runs projection matching per candidate and keeps the one whose final pass has the highest mean correlation
    between images and reprojections (on 00254 this peaks exactly where the bead residuals are lowest), then
    tries +-refine_step around it. Returns (thickness, its ProjectionMatching, [(thickness, score), ...])."""
    if candidates_angstrom is None:
        candidates_angstrom = [1500, 2000, 2500, 3000, 4000, 5000]
    pm_kwargs.setdefault("verbose", False)
    results: dict[float, ProjectionMatching] = {}

    def run(t):
        if t not in results:
            results[t] = refine_by_projection_matching(ts, images, t, **pm_kwargs)
        return results[t].history[-1]["mean_cc"]

    best = max(candidates_angstrom, key=run)
    if refine_step_angstrom:
        for t in (best - refine_step_angstrom, best + refine_step_angstrom):
            if t > 0:
                run(t)
        best = max(results, key=lambda t: results[t].history[-1]["mean_cc"])
    table = sorted((t, r.history[-1]["mean_cc"]) for t, r in results.items())
    return best, results[best], table


# ---------------------------------------------------------------------------------------------- CTF


def warp_average_paths(tilt_series_xml: Path | str, frameseries_dir: Path | str) -> list[Path]:
    """Full-size motion-corrected averages in *stack order*, from a Warp tilt-series XML's MoviePath list.
    (Warp names movies with the acquisition's angle sign, which can be opposite to the XML's: go by order.)"""
    s = Path(tilt_series_xml).read_text()
    block = s[s.index("<MoviePath>") + len("<MoviePath>"):s.index("</MoviePath>")]
    names = [Path(line.strip()).stem for line in block.splitlines() if line.strip()]
    return [Path(frameseries_dir) / "average" / f"{n}.mrc" for n in names]


def estimate_ctf_per_tilt(
    image_paths: list[Path],
    pixel_spacing: float,
    voltage_kv: float = 300.0,
    spherical_aberration_mm: float = 2.7,
    amplitude_contrast: float = 0.07,
    fit_range_angstrom: tuple[float, float] = (30.0, 8.0),
    defocus_range_um: tuple[float, float] = (0.5, 6.0),
    min_cc: float = 0.01,
    verbose: bool = True,
) -> list[dict]:
    """torch_ctf_estimation.estimate_ctf on every tilt image (full pixel size: binned stacks have no Thon rings).

    Linear defocus model per image: mean defocus + astigmatism, plus the defocus gradient, whose magnitude is the
    specimen tilt seen by the CTF alone (a check on the tilt angles). `reliable` = 2-D fit correlation >= min_cc."""
    from torch_ctf_estimation import CTFFittingParams, OpticalParams, estimate_ctf

    optics = OpticalParams(pixel_spacing_angstroms=pixel_spacing, voltage_kev=voltage_kv,
                           spherical_aberration_mm=spherical_aberration_mm,
                           amplitude_contrast_fraction=amplitude_contrast)
    fitting = CTFFittingParams(defocus_grid_resolution=(1, 2, 2), frequency_fit_range_angstroms=fit_range_angstrom,
                               defocus_range_microns=defocus_range_um, defocus_model="linear")
    rows, t0 = [], time.time()
    for i, path in enumerate(image_paths):
        with mrcfile.mmap(path, permissive=True) as m:
            img = torch.tensor(np.asarray(m.data, dtype=np.float32))
        _, r1, r2 = estimate_ctf(img, optics, fitting)
        _, tilt_mag = r2.linear_tilt_axis_and_magnitude_deg(pixel_spacing, max(img.shape))
        rows.append({"index": i, "image": Path(path).name,
                     "defocus_um": 0.5 * (r2.defocus_u + r2.defocus_v),
                     "astigmatism_um": 0.5 * (r2.defocus_u - r2.defocus_v),
                     "astigmatism_angle": float(r2.astigmatism_angle or 0.0),
                     "defocus_1d_um": float(r1.ctf_model.defocus_um),
                     "ctf_tilt_deg": float(tilt_mag), "cc": float(r2.cross_correlation_final),
                     "reliable": float(r2.cross_correlation_final) >= min_cc})
        if verbose and (i % 10 == 0 or i == len(image_paths) - 1):
            print(f"CTF {i + 1}/{len(image_paths)} ({time.time() - t0:.0f} s)")
    return rows


def warp_defocus(frameseries_dir: Path | str, image_paths: list[Path]) -> np.ndarray:
    """Warp's per-movie defocus (um) for the same images, from <frameseries>/<name>.xml (validation only)."""
    import re

    out = []
    for p in image_paths:
        s = (Path(frameseries_dir) / f"{Path(p).stem}.xml").read_text()
        out.append(float(re.search(r'Name="Defocus" Value="([^"]+)"', s[s.index("<CTF>"):]).group(1)))
    return np.array(out)


def phase_flip(stack: torch.Tensor, pixel_spacing: float, ctf_rows: list[dict], voltage_kv: float = 300.0,
               spherical_aberration_mm: float = 2.7, amplitude_contrast: float = 0.07) -> torch.Tensor:
    """Multiply every tilt's spectrum by the sign of its CTF (one defocus per tilt; the defocus gradient across a
    tilted image is ignored — fine when the first CTF zero is not far inside Nyquist)."""
    from torch_ctf import calculate_ctf_2d

    n, h, w = stack.shape
    dev = stack.device
    ctf = calculate_ctf_2d(
        defocus=torch.tensor([r["defocus_um"] for r in ctf_rows], device=dev),
        astigmatism=torch.tensor([r["astigmatism_um"] for r in ctf_rows], device=dev),
        astigmatism_angle=torch.tensor([r["astigmatism_angle"] for r in ctf_rows], device=dev),
        voltage=voltage_kv, spherical_aberration=spherical_aberration_mm, amplitude_contrast=amplitude_contrast,
        phase_shift=0.0, pixel_size=pixel_spacing, image_shape=(h, w), rfft=True, fftshift=False)
    return torch.fft.irfft2(torch.fft.rfft2(stack) * torch.sign(ctf), s=(h, w))


def first_ctf_zero_angstrom(defocus_um: float, voltage_kv: float = 300.0, spherical_aberration_mm: float = 2.7,
                            amplitude_contrast: float = 0.07) -> float:
    """Resolution (A) of the first CTF zero, from the 1-D CTF (to say whether phase flipping matters at a pixel size)."""
    from torch_ctf import calculate_ctf_1d

    c = calculate_ctf_1d(defocus=defocus_um, voltage=voltage_kv, spherical_aberration=spherical_aberration_mm,
                         amplitude_contrast=amplitude_contrast, phase_shift=0.0, pixel_size=1.0, n_samples=40000,
                         oversampling_factor=1)
    k = torch.linspace(0, 0.5, len(c))
    sign_change = torch.nonzero(torch.sign(c[1:]) != torch.sign(c[:-1]))
    return float(1.0 / k[int(sign_change[0]) + 1]) if len(sign_change) else float("nan")


# ---------------------------------------------------------------------------------------------- output


def save_alignment(ts: TiltSeries, path: Path | str, **extra) -> None:
    """The alignment as JSON (Angstrom / degrees): tilt angles, axis angle, sample translations (y, x), + extra."""
    import json

    out = {"tilt_angles_deg": ts.tilt_angles.cpu().tolist(), "tilt_axis_angle_deg": ts.tilt_axis_angle.cpu().tolist(),
           "sample_translations_yx_angstrom": ts.sample_translations.cpu().tolist(),
           "pixel_spacing_angstrom": ts.pixel_spacing, "image_path": str(ts.image_path) if ts.image_path else None,
           **extra}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))


def load_alignment(path: Path | str, device: str = "cpu") -> TiltSeries:
    """Inverse of save_alignment."""
    import json

    d = json.loads(Path(path).read_text())
    return TiltSeries(tilt_angles=torch.tensor(d["tilt_angles_deg"]), tilt_axis_angle=torch.tensor(d["tilt_axis_angle_deg"]),
                      sample_translations=torch.tensor(d["sample_translations_yx_angstrom"]),
                      image_path=d["image_path"], pixel_spacing=d["pixel_spacing_angstrom"], device=device)
