"""Warp-compatible mode for TeamTomo's reconstruction: torch-reconstruct-tomogram, set up to reproduce WarpTools
`ts_reconstruct` on the same alignment. Pure TeamTomo + torch (no warpylib), so it runs in every demo venv.

Five differences between `reconstruct_tomogram` (as phase 1/2 use it) and Warp, each measured on 00254 against the real
Warp volume (phase2b notebook, section 4b), each mirroring Warp 5ffe603 / 2.0.0+a748f1e2:

1. **Image pixel size.** Fourier-cropping 3838 x 3710 px at 1.7005 A to whole pixels (652 x 630) makes the true spacing
   6526.5/652 = 10.0100 A (x) and 6308.9/630 = 10.0141 A (y), not the 10.0 in the stack header. Warp corrects for it
   (`SizeRoundingFactors`, TiltSeries.cs L1505 and L513); TeamTomo takes the header. 0.1 % scale -> up to 0.6 px at the edge.
2. **Volume centre.** Warp centres the tomogram at `DimensionsPhysical / 2` (TiltSeries.cs `VolumeCenter`), i.e.
   voxel 326.33 / 315.44 / 170.05 for 00254; `reconstruct_tomogram` at `shape // 2`. Constant 0.3-0.4 voxel offset.
3. **Preprocessing** (ReconstructFull.cs L76-90): plane subtraction, 16-px cosine edge inside the image, sharp high-pass
   at 1/(SubVolumeSize * SubVolumePadding / 2) Nyquist = 1/192 cycles/px, normalisation. TeamTomo removes only DC.
4. **CTF and dose** (ReconstructFull.cs L164-171, CTF.cuh): each image x CTF x cos(tilt) x exp(-Dose s^2), divided by
   the unweighted |CTF|. Here per tilt (one defocus per tilt, no depth gradient): sign(CTF) x cos(tilt) x exp(-Dose s^2).

5. **Sampling weights** (ReconstructFull.cs L182-188: data x min(sampling, 1) / max(weights, 0.01), "We don't want to
   bring voxels with interpolation weight < 1 back up"). `reconstruct_tomogram` divides every Fourier voxel by its
   insertion weight, so voxels between the tilt slices (weight < 1) are boosted: 1.5-2x more amplitude at 100-20 A than
   Warp, which reads as a noisier, flatter tomogram even at 0.96 correlation. Here: divide by max(weight, 1), applied
   as a runtime patch of torch_reconstruct_tomogram's per-patch normalisation for the duration of the call.

Results on 00254 (IMOD alignment): corr vs real Warp at full resolution 0.655 (default) -> 0.959 (points 1-4)
-> 0.995 (all five; 0.999 at sigma 2 voxels, amplitude ratio 0.999, radial spectrum within 2 % of Warp's).
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import textwrap

import numpy as np
import torch


@contextlib.contextmanager
def warp_sampling_weights():
    """Temporarily make torch_reconstruct_tomogram normalise like Warp: divide by max(weight, 1) (point 5)."""
    import torch_reconstruct_tomogram.reconstruct as R

    src = textwrap.dedent(inspect.getsource(R._reconstruct_subvolume))
    old = "    valid_weights = weights > 1e-3\n    patches_rfft[:, valid_weights] /= weights[valid_weights]\n"
    if src.count(old) != 1:
        raise RuntimeError("torch_reconstruct_tomogram changed: re-check warp_compat's sampling-weight patch")
    new = "    patches_rfft = patches_rfft / torch.clamp(weights, min=1.0)   # Warp: no up-weighting of partial samples\n"
    ns = dict(vars(R))
    exec(compile(src.replace(old, new), R.__file__, "exec"), ns)
    original = R._reconstruct_subvolume
    R._reconstruct_subvolume = ns["_reconstruct_subvolume"]
    try:
        yield
    finally:
        R._reconstruct_subvolume = original


def true_image_spacing(image_dimensions_angstrom_xy, image_shape_hw) -> tuple[float, float]:
    """(y, x) spacing of a stack whose physical extent is known (Warp XML `ImageDimensionsAngstrom`)."""
    (dx, dy), (h, w) = image_dimensions_angstrom_xy, image_shape_hw
    return dy / h, dx / w


def warp_geometry(ts, volume_dimensions_angstrom_xyz, volume_shape, voxel_size: float, image_spacing_yx):
    """Copy of `ts` whose image pixel size and tomogram centre follow Warp (points 1 and 2 above).

    The images keep their shape; `pixel_spacing` becomes their true (mean) spacing, and the reconstruction must be run
    with output_pixel_spacing=voxel_size. The y/x anisotropy (0.04 % for 00254) is averaged."""
    t = copy.copy(ts)
    t.pixel_spacing = float(np.mean(image_spacing_yx))
    centre = np.asarray(volume_dimensions_angstrom_xyz, dtype=np.float64)[::-1] / 2      # zyx, Warp's VolumeCenter
    shift = centre - (np.asarray(volume_shape) // 2) * voxel_size                         # vs reconstruct_tomogram's
    M = ts.levelled2tomo.clone()
    M[:3, 3] += torch.tensor(shift, dtype=M.dtype, device=M.device)
    t.levelled2tomo = M
    return t


def warp_preprocess(images: torch.Tensor, subvolume_size: int = 64, subvolume_padding: float = 3.0) -> torch.Tensor:
    """ReconstructFull.cs L76-90 with Normalize on: plane fit/subtract, MaskRectangularly(dims-32, 16), sharp
    Bandpass(1/(size*padding/2) of Nyquist, 1), normalise to mean 0 / std 1. No inversion (TeamTomo's polarity)."""
    n, h, w = images.shape
    dev = images.device
    yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32), torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    A = torch.stack([torch.ones_like(xx).flatten(), xx.flatten(), yy.flatten()], 1)
    coef = torch.linalg.lstsq(A, images.reshape(n, -1).T).solution                       # (3, n)
    out = images - (A @ coef).T.reshape(n, h, w)

    def edge(m):
        x = torch.arange(m, device=dev, dtype=torch.float32)
        d = torch.abs(x - m / 2) - (m - 32) / 2                                             # distance beyond the flat region
        return torch.where(d <= 0, torch.ones_like(d), 0.5 + 0.5 * torch.cos(torch.pi * torch.clamp(d / 16, 0, 1)))
    out = out * (edge(h)[:, None] * edge(w)[None, :])

    ky = torch.fft.fftfreq(h, device=dev)[:, None]
    kx = torch.fft.rfftfreq(w, device=dev)[None, :]
    r_nyquist = torch.sqrt(ky ** 2 + kx ** 2) / 0.5
    keep = (r_nyquist >= 1.0 / (subvolume_size * subvolume_padding / 2)) & (r_nyquist <= 1.0)
    out = torch.fft.irfft2(torch.fft.rfft2(out) * keep, s=(h, w))
    out = out - out.mean(dim=(-2, -1), keepdim=True)
    return out / out.std(dim=(-2, -1), keepdim=True)


def warp_ctf_and_dose(images: torch.Tensor, pixel_spacing: float, tilt_angles, dose, defocus_um, voltage_kv: float = 300.0,
                      spherical_aberration_mm: float = 2.7, amplitude_contrast: float = 0.07, astigmatism_um=None,
                      astigmatism_angle=None) -> torch.Tensor:
    """Per tilt: sign(CTF) x cos(tilt) x exp(-Dose * s^2), Warp's weighting with trivial dose grids (CTF.cuh L76-100;
    TiltSeries.cs GetCTFsForOneParticle: scale cos(tilt), B = -4 Dose)."""
    from torch_ctf import calculate_ctf_2d

    n, h, w = images.shape
    dev = images.device
    t = lambda v, default=0.0: torch.as_tensor(np.full(n, default) if v is None else np.asarray(v), dtype=torch.float32, device=dev)
    ctf = calculate_ctf_2d(defocus=t(defocus_um), astigmatism=t(astigmatism_um), astigmatism_angle=t(astigmatism_angle),
                           voltage=voltage_kv, spherical_aberration=spherical_aberration_mm,
                           amplitude_contrast=amplitude_contrast, phase_shift=0.0, pixel_size=pixel_spacing,
                           image_shape=(h, w), rfft=True, fftshift=False)
    ky = torch.fft.fftfreq(h, d=pixel_spacing, device=dev)[:, None]
    kx = torch.fft.rfftfreq(w, d=pixel_spacing, device=dev)[None, :]
    weight = torch.cos(torch.deg2rad(t(tilt_angles)))[:, None, None] * torch.exp(-t(dose)[:, None, None] * (ky ** 2 + kx ** 2)[None])
    return torch.fft.irfft2(torch.fft.rfft2(images) * torch.sign(ctf) * weight, s=(h, w))


def reconstruct_warp_compatible(ts, images: torch.Tensor, volume_dimensions_angstrom_xyz, image_dimensions_angstrom_xy,
                                voxel_size: float, dose=None, defocus_um=None, ctf_and_dose: bool = True,
                                sampling_weights: bool = True, sidelength: int = 128, batch_size: int = 4,
                                **ctf_kwargs) -> torch.Tensor:
    """reconstruct_tomogram with all four Warp conventions (see module docstring); returns (z, y, x) at voxel_size.

    `images` are the raw tilt images (any pixel size whose physical extent is image_dimensions_angstrom_xy)."""
    from torch_reconstruct_tomogram import reconstruct_tomogram

    shape = tuple(int(round(d / voxel_size / 2)) * 2 for d in volume_dimensions_angstrom_xyz)[::-1]   # Warp's rounding
    spacing = true_image_spacing(image_dimensions_angstrom_xy, images.shape[-2:])
    t = warp_geometry(ts, volume_dimensions_angstrom_xyz, shape, voxel_size, spacing)
    t.to(images.device)                       # the copy only: the caller's TiltSeries stays where it is
    img = warp_preprocess(images)
    if ctf_and_dose:
        img = warp_ctf_and_dose(img, t.pixel_spacing, ts.tilt_angles.cpu().numpy(), dose, defocus_um, **ctf_kwargs)
    with (warp_sampling_weights() if sampling_weights else contextlib.nullcontext()):
        return reconstruct_tomogram(t, volume_shape=shape, sidelength=sidelength, batch_size=batch_size, images=img,
                                    output_pixel_spacing=voxel_size, preprocess=False)


def warp_xml_parameters(xml_path) -> dict:
    """dose, defocus and image/volume dimensions from a Warp tilt-series XML (for reconstruct_warp_compatible)."""
    import warpxmlfile

    x = warpxmlfile.read(xml_path)
    return {"dose": np.asarray(x.dose, dtype=np.float64),
            "defocus_um": np.asarray(x.grids["GridCTF"].values, dtype=np.float64).reshape(-1),
            "image_dimensions_angstrom_xy": tuple(float(v) for v in x.image_dimensions_angstrom),
            "volume_dimensions_angstrom_xyz": tuple(float(v) for v in x.volume_dimensions_angstrom)}
