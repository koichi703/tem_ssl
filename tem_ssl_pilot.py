#!/usr/bin/env python3
"""
TEM self-supervised-learning pilot pipeline.

Commands
--------
prepare:
    DM3/DM4/TIFF/etc. -> physical-scale-aware patches + manifest.

train:
    SimCLR-style self-supervised training with source-file grouped splits.

extract:
    Extract learned 512-D embeddings and handcrafted TEM features.

analyze:
    Feature selection for handcrafted descriptors, PCA, simple clustering,
    and cross-source nearest-neighbour checks.

The design intentionally keeps different physical-scale groups separate so that a
model does not simply learn microscope magnification / pixel-size differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

SUPPORTED = {
    ".dm3", ".dm4", ".emd", ".ser",
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"
}


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "item"


def sha1_short(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def robust_normalize(
    image: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("Image has no finite pixels.")
    fill = float(np.nanmedian(x[finite]))
    x = np.where(finite, x, fill)

    lo, hi = np.percentile(x, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)


def unit_scale_to_nm(scale: float, units: Any) -> float | None:
    if scale is None:
        return None
    try:
        value = abs(float(scale))
    except Exception:
        return None

    u = "" if units is None else str(units).strip().lower()
    u = u.replace("μ", "µ").replace("ångström", "angstrom")

    factors = {
        "pm": 1e-3,
        "nm": 1.0,
        "um": 1e3,
        "µm": 1e3,
        "micrometer": 1e3,
        "micrometre": 1e3,
        "mm": 1e6,
        "m": 1e9,
        "å": 0.1,
        "a": 0.1,
        "angstrom": 0.1,
        "ang": 0.1,
    }

    if u in factors:
        return value * factors[u]

    # Some readers return strings such as "nm/pixel".
    for key, factor in factors.items():
        if u.startswith(key + "/") or u.startswith(key + " "):
            return value * factor

    return None


def flatten_dict(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_dict(v, p))
    elif isinstance(obj, (list, tuple)):
        # Avoid exploding large arrays/lists.
        if len(obj) <= 20:
            for i, v in enumerate(obj):
                p = f"{prefix}[{i}]"
                out.update(flatten_dict(v, p))
        else:
            out[prefix] = f"<list length={len(obj)}>"
    else:
        if isinstance(obj, np.generic):
            obj = obj.item()
        out[prefix] = obj
    return out


def metadata_find(flat: dict[str, Any], needles: list[str]) -> Any:
    needles = [n.lower() for n in needles]
    candidates = []
    for k, v in flat.items():
        kl = k.lower()
        if all(n in kl for n in needles):
            candidates.append((len(k), k, v))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][2]


@dataclass
class SourceImage:
    source_file: Path
    signal_index: int
    image: np.ndarray
    pixel_size_nm_x: float | None
    pixel_size_nm_y: float | None
    metadata: dict[str, Any]
    original_metadata: dict[str, Any]


def _extract_signal_axis_scales_nm(sig: Any) -> tuple[float | None, float | None]:
    try:
        axes = list(sig.axes_manager.signal_axes)
    except Exception:
        axes = []

    if len(axes) < 2:
        try:
            axes = list(sig.axes_manager)[-2:]
        except Exception:
            axes = []

    vals = []
    for ax in axes[:2]:
        nm = unit_scale_to_nm(getattr(ax, "scale", None), getattr(ax, "units", None))
        vals.append(nm)

    if len(vals) >= 2:
        # For 2D data HyperSpy often exposes x/y signal axes; exact order is not
        # critical here because we retain both and use their mean for square crops.
        return vals[0], vals[1]
    return None, None


def load_source_images(
    path: Path,
    fallback_nm_per_pixel: float | None = None,
) -> list[SourceImage]:
    suffix = path.suffix.lower()

    if suffix in {".dm3", ".dm4", ".emd", ".ser"}:
        try:
            import hyperspy.api as hs
        except ImportError as e:
            raise RuntimeError(
                f"{path.name}: HyperSpy is required for {suffix}. "
                "Install with: pip install hyperspy"
            ) from e

        loaded = hs.load(str(path), lazy=False)
        signals = loaded if isinstance(loaded, list) else [loaded]
        result = []

        for signal_index, sig in enumerate(signals):
            arr = np.squeeze(np.asarray(sig.data))
            if arr.ndim != 2:
                continue

            sx, sy = _extract_signal_axis_scales_nm(sig)
            if sx is None:
                sx = fallback_nm_per_pixel
            if sy is None:
                sy = fallback_nm_per_pixel

            try:
                md = sig.metadata.as_dictionary()
            except Exception:
                md = {}
            try:
                omd = sig.original_metadata.as_dictionary()
            except Exception:
                omd = {}

            result.append(
                SourceImage(
                    source_file=path,
                    signal_index=signal_index,
                    image=arr.astype(np.float32, copy=False),
                    pixel_size_nm_x=sx,
                    pixel_size_nm_y=sy,
                    metadata=md,
                    original_metadata=omd,
                )
            )

        if not result:
            raise ValueError(f"{path.name}: no readable 2D signal.")
        return result

    # Ordinary raster images. Physical scale is unknown unless provided manually.
    from skimage import color, io

    arr = np.asarray(io.imread(path))
    arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = color.rgb2gray(arr[..., :3])

    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D raster image, got {arr.shape}")

    return [
        SourceImage(
            source_file=path,
            signal_index=0,
            image=arr.astype(np.float32, copy=False),
            pixel_size_nm_x=fallback_nm_per_pixel,
            pixel_size_nm_y=fallback_nm_per_pixel,
            metadata={},
            original_metadata={},
        )
    ]


DEFAULT_BRAGG_D_MIN_NM = 0.12
DEFAULT_BRAGG_D_MAX_NM = 1.0


def azimuthal_bragg_score(
    arr01: np.ndarray,
    patch_fov_nm: float,
    d_min_nm: float = DEFAULT_BRAGG_D_MIN_NM,
    d_max_nm: float = DEFAULT_BRAGG_D_MAX_NM,
    patch_fov_nm_y: float | None = None,
    source_pixels: int | None = None,
) -> dict[str, float]:
    """
    Separate discrete Bragg reflections from the diffuse amorphous halo.

    A crystalline patch concentrates power at a few azimuths on one radial
    ring, so max/median around that ring is large. An amorphous halo is flat
    in azimuth and its ratio stays near unity. Because patches are cropped by
    physical field of view, the frequency axis is in nm^-1 and the score is
    directly comparable across sources within a scale group.

    The band is clamped to 90% of the patch Nyquist frequency, so a coarse
    group (meso/micro) whose pixels cannot resolve d_min_nm reports no score
    at all (NaN in every field) rather than measuring noise or being mistaken
    for a genuinely low, measured value.
    """
    nan = float("nan")
    empty = {"bragg_ratio": nan, "bragg_d_nm": nan, "bragg_pixels": nan}

    n = int(arr01.shape[0])
    if arr01.ndim != 2 or arr01.shape[0] != arr01.shape[1] or n < 16:
        return empty
    if not np.isfinite(patch_fov_nm) or patch_fov_nm <= 0:
        return empty
    if d_min_nm <= 0 or d_max_nm <= d_min_nm:
        return empty

    # A source with non-square physical pixels covers different physical
    # widths along x and y even though the patch is square in pixels. Sharing
    # one frequency axis would stretch a physically circular amorphous halo
    # into an ellipse, and the azimuthal max/median ratio would read that
    # ellipse as a Bragg reflection.
    fov_x = float(patch_fov_nm)
    fov_y = fov_x if patch_fov_nm_y is None else float(patch_fov_nm_y)
    if not np.isfinite(fov_y) or fov_y <= 0:
        fov_y = fov_x

    x = np.asarray(arr01, dtype=np.float64)
    x = x - x.mean()
    window = np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fftshift(np.fft.fft2(x * window))) ** 2

    freq_x = np.fft.fftshift(np.fft.fftfreq(n, d=fov_x / n))
    freq_y = np.fft.fftshift(np.fft.fftfreq(n, d=fov_y / n))
    fx, fy = np.meshgrid(freq_x, freq_y)
    radius = np.sqrt(fx ** 2 + fy ** 2)

    # Resampling to output_pixels cannot add information. When the source crop
    # is smaller than the patch grid the image was upsampled, so scoring up to
    # the patch Nyquist would let interpolation ringing be picked as a peak;
    # the acquisition grid sets the real limit.
    grid_px = n if source_pixels is None else min(n, int(source_pixels))
    nyquist = grid_px / (2.0 * max(fov_x, fov_y))
    f_lo = 1.0 / d_max_nm
    f_hi = min(1.0 / d_min_nm, 0.9 * nyquist)
    # A band clamped down to a hair above f_lo spans a single radial ring, so
    # max/median there only measures noise. Require real width instead, which
    # is how a coarse group whose pixels cannot resolve lattice fringes ends
    # up reporting no score at all.
    if f_hi < 1.5 * f_lo:
        return empty

    # Coarser of the two frequency steps, so each ring keeps enough samples
    # for a robust azimuthal median; identical to 1/FOV when pixels are square.
    bin_width = 1.0 / min(fov_x, fov_y)
    rbin = np.rint(radius / bin_width).astype(int)
    band = (radius >= f_lo) & (radius <= f_hi)
    if not band.any():
        return empty

    ring_stats = []
    for b in np.unique(rbin[band]):
        ring = band & (rbin == b)
        # Too few samples to estimate an azimuthal median robustly.
        if ring.sum() < 12:
            continue
        vals = power[ring]
        median = float(np.median(vals))
        if not np.isfinite(median) or median <= 0:
            continue
        ring_stats.append((b, vals, median, int(ring.sum()), radius[ring]))

    if len(ring_stats) < 2:
        return empty

    # A ring carrying a vanishing fraction of the patch's power holds only
    # window sidelobes and float rounding, where max/median explodes to 1e9
    # while meaning nothing -- a pure sinusoid would otherwise be reported at
    # an empty high-frequency ring instead of its own.
    #
    # The test is on the ring's peak, not its median: a sharp reflection puts
    # two bright pixels on a ring of fifty, so the very rings that matter most
    # have a near-zero median and a median-based floor would discard exactly
    # them. Real patches span ~1e3 between their strongest and weakest ring,
    # so 1e6 below the band peak is far outside genuine signal.
    band_peak = max(float(stat[1].max()) for stat in ring_stats)
    floor = band_peak * 1e-6

    best_ratio = 0.0
    best_bin = None
    peak_pixels = 0
    scored = []
    for b, vals, median, count, ring_radius in ring_stats:
        if float(vals.max()) < floor:
            continue
        scored.append((b, vals, ring_radius))
        peak_pixels += int((vals > 5.0 * median).sum())

        # Power on a noise-only ring is roughly exponential, so max/median
        # already grows like log(count) with the ring's circumference. Divide
        # that expectation out so rings of different radius are comparable and
        # an isotropic halo sits near 1 instead of near 10.
        expected_max = (np.log(count) + np.euler_gamma) / np.log(2.0)
        ratio = (float(vals.max()) / median) / float(expected_max)
        if ratio > best_ratio:
            best_ratio = ratio
            best_bin = b

    if len(scored) < 2 or best_bin is None:
        return empty

    # The spacing must describe the reflection the score actually fired on.
    # Taking the brightest pixel of the whole band instead would report
    # low-frequency morphology whenever that carries more absolute power than
    # a weak lattice reflection. Neighbouring bins are included because
    # leakage raises the true ring's own median, which can push the ratio one
    # bin off the reflection.
    peak_power = -np.inf
    peak_radius = 0.0
    for b, vals, ring_radius in scored:
        if abs(int(b) - int(best_bin)) > 1:
            continue
        brightest = int(np.argmax(vals))
        if float(vals[brightest]) > peak_power:
            peak_power = float(vals[brightest])
            peak_radius = float(ring_radius[brightest])

    return {
        "bragg_ratio": best_ratio,
        "bragg_d_nm": 1.0 / peak_radius if peak_radius > 0 else nan,
        "bragg_pixels": float(peak_pixels),
    }


def choose_scale_group(nm_per_pixel: float, config: dict[str, Any]) -> dict[str, Any]:
    for group in config["scale_groups"]:
        maxv = group.get("max_nm_per_pixel")
        if maxv is None or nm_per_pixel <= float(maxv):
            return group
    return config["scale_groups"][-1]


def grid_positions(
    height: int,
    width: int,
    crop: int,
    overlap: float,
) -> list[tuple[int, int]]:
    if crop > height or crop > width:
        return []

    stride = max(1, int(round(crop * (1.0 - overlap))))
    ys = list(range(0, height - crop + 1, stride))
    xs = list(range(0, width - crop + 1, stride))

    if ys and ys[-1] != height - crop:
        ys.append(height - crop)
    if xs and xs[-1] != width - crop:
        xs.append(width - crop)

    return [(y, x) for y in ys for x in xs]


# ---------------------------------------------------------------------
# PREPARE
# ---------------------------------------------------------------------
def command_prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    input_dir = args.input_dir.resolve()
    project = args.project_dir.resolve()
    dataset_dir = project / "dataset"
    patch_root = dataset_dir / "patches"
    metadata_root = dataset_dir / "metadata"
    patch_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not files:
        raise RuntimeError(f"No supported images found in {input_dir}")

    clip_low, clip_high = config.get("clip_percentiles", [0.5, 99.5])
    output_pixels = int(config.get("output_pixels", 224))
    overlap = float(config.get("overlap", 0.5))
    max_patches = int(config.get("max_patches_per_source", 250))
    min_crop_pixels = int(config.get("min_crop_pixels", 64))

    rows = []
    source_rows = []
    status_rows = []

    for file_no, path in enumerate(files, start=1):
        print(f"[{file_no}/{len(files)}] {path.name}")
        try:
            sources = load_source_images(
                path,
                fallback_nm_per_pixel=args.fallback_nm_per_pixel,
            )
        except Exception as e:
            warnings.warn(f"Could not load {path}: {e}")
            status_rows.append({
                "source_file": str(path),
                "signal_index": "",
                "status": "load_failed",
                "message": str(e),
            })
            continue

        for src in sources:
            source_key = (
                f"{path.stem}"
                if len(sources) == 1
                else f"{path.stem}__signal{src.signal_index:02d}"
            )

            sx, sy = src.pixel_size_nm_x, src.pixel_size_nm_y
            if sx is None or sy is None:
                msg = "Physical pixel size is missing."
                warnings.warn(f"{path.name}: {msg}")
                status_rows.append({
                    "source_file": str(path),
                    "signal_index": src.signal_index,
                    "status": "missing_scale",
                    "message": msg,
                })
                continue

            nm_per_pixel = float((sx + sy) / 2.0)
            group = choose_scale_group(nm_per_pixel, config)
            group_name = str(group["name"])
            patch_fov_nm = float(group["patch_fov_nm"])
            crop_px = int(round(patch_fov_nm / nm_per_pixel))

            if crop_px < min_crop_pixels:
                msg = (
                    f"Physical patch would be only {crop_px}px; "
                    f"minimum is {min_crop_pixels}px."
                )
                warnings.warn(f"{path.name}: {msg}")
                status_rows.append({
                    "source_file": str(path),
                    "signal_index": src.signal_index,
                    "status": "crop_too_small",
                    "message": msg,
                })
                continue

            h, w = src.image.shape
            positions = grid_positions(h, w, crop_px, overlap)
            if not positions:
                msg = f"Requested crop {crop_px}px exceeds image {w}x{h}px."
                warnings.warn(f"{path.name}: {msg}")
                status_rows.append({
                    "source_file": str(path),
                    "signal_index": src.signal_index,
                    "status": "crop_too_large",
                    "message": msg,
                })
                continue

            # Deterministic balancing: do not allow one large-FOV source to dominate.
            if len(positions) > max_patches:
                local_seed = seed + int(sha1_short(str(path), 8), 16) % 1_000_000
                rng = random.Random(local_seed)
                positions = sorted(rng.sample(positions, max_patches))

            flat_md = flatten_dict(src.metadata)
            flat_omd = flatten_dict(src.original_metadata)
            combined = {**{f"metadata.{k}": v for k, v in flat_md.items()},
                        **{f"original_metadata.{k}": v for k, v in flat_omd.items()}}

            magnification = metadata_find(combined, ["magnification"])
            beam_energy = (
                metadata_find(combined, ["beam", "energy"])
                or metadata_find(combined, ["voltage"])
                or metadata_find(combined, ["accelerating"])
            )
            microscope = (
                metadata_find(combined, ["microscope"])
                or metadata_find(combined, ["instrument"])
            )

            source_id = safe_name(source_key) + "__" + sha1_short(str(path))
            metadata_path = metadata_root / f"{source_id}.json"
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_file": str(path),
                        "signal_index": src.signal_index,
                        "pixel_size_nm_x": sx,
                        "pixel_size_nm_y": sy,
                        "metadata": src.metadata,
                        "original_metadata": src.original_metadata,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            source_rows.append({
                "source_id": source_id,
                "source_file": str(path),
                "signal_index": src.signal_index,
                "height_px": h,
                "width_px": w,
                "pixel_size_nm_x": sx,
                "pixel_size_nm_y": sy,
                "nm_per_pixel_mean": nm_per_pixel,
                "field_of_view_nm_x": w * sx,
                "field_of_view_nm_y": h * sy,
                "scale_group": group_name,
                "patch_fov_nm": patch_fov_nm,
                "crop_size_px": crop_px,
                "patch_count": len(positions),
                "magnification": magnification,
                "beam_energy_or_voltage": beam_energy,
                "microscope": microscope,
                "metadata_json": str(metadata_path.relative_to(project)),
            })

            norm = robust_normalize(src.image, clip_low, clip_high)
            out_dir = patch_root / group_name / source_id
            out_dir.mkdir(parents=True, exist_ok=True)

            for patch_no, (y, x) in enumerate(positions):
                crop = norm[y:y + crop_px, x:x + crop_px]

                pil = Image.fromarray(np.clip(crop * 255.0, 0, 255).astype(np.uint8))
                pil = pil.resize(
                    (output_pixels, output_pixels),
                    resample=Image.Resampling.LANCZOS,
                )

                patch_id = f"{source_id}__p{patch_no:04d}"
                patch_path = out_dir / f"{patch_id}.png"
                pil.save(patch_path)

                # Scored on the resampled patch, because the 224 px grid is
                # what maps onto patch_fov_nm.
                bragg = azimuthal_bragg_score(
                    np.asarray(pil, dtype=np.float32) / 255.0,
                    crop_px * sx,
                    patch_fov_nm_y=crop_px * sy,
                    source_pixels=crop_px,
                )

                rows.append({
                    "patch_id": patch_id,
                    "patch_path": str(patch_path.relative_to(project)),
                    "source_id": source_id,
                    "source_file": str(path),
                    "signal_index": src.signal_index,
                    "scale_group": group_name,
                    "nm_per_pixel_source": nm_per_pixel,
                    "patch_fov_nm": patch_fov_nm,
                    "patch_fov_nm_x": crop_px * sx,
                    "patch_fov_nm_y": crop_px * sy,
                    "crop_size_px_source": crop_px,
                    "output_pixels": output_pixels,
                    "x_px_source": x,
                    "y_px_source": y,
                    "x_nm_source": x * sx,
                    "y_nm_source": y * sy,
                    "patch_mean_normalized": float(np.mean(crop)),
                    "patch_std_normalized": float(np.std(crop)),
                    **bragg,
                })

            status_rows.append({
                "source_file": str(path),
                "signal_index": src.signal_index,
                "status": "ok",
                "message": f"{len(positions)} patches, group={group_name}",
            })

    if not rows:
        raise RuntimeError(
            "No patches were produced. Check physical pixel-size metadata and config."
        )

    manifest = pd.DataFrame(rows)
    sources_df = pd.DataFrame(source_rows)
    status_df = pd.DataFrame(status_rows)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(dataset_dir / "manifest.csv", index=False)
    sources_df.to_csv(dataset_dir / "sources.csv", index=False)
    status_df.to_csv(dataset_dir / "prepare_status.csv", index=False)
    config_used_path = project / "pilot_config_used.json"
    config_used_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nPrepared pilot dataset.")
    print(f"Sources successfully prepared : {sources_df['source_id'].nunique()}")
    print(f"Patches                      : {len(manifest)}")
    print("\nPatches by scale group:")
    print(manifest.groupby("scale_group").size().to_string())
    print(f"\nProject: {project}")


# ---------------------------------------------------------------------
# SimCLR model / dataset
# ---------------------------------------------------------------------
def build_model(projection_dim: int = 128):
    import torch.nn as nn
    from torchvision.models import resnet18

    class SimCLRModel(nn.Module):
        def __init__(self):
            super().__init__()
            encoder = resnet18(weights=None)
            encoder.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            dim = encoder.fc.in_features
            encoder.fc = nn.Identity()
            self.encoder = encoder
            self.projector = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, projection_dim),
            )
            self.embedding_dim = dim

        def forward(self, x):
            h = self.encoder(x)
            z = self.projector(h)
            return h, z

    return SimCLRModel()


class PatchDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        project_dir: Path,
        pair_views: bool,
        augment: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.project_dir = project_dir
        self.pair_views = pair_views
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_tensor(self, idx: int):
        import torch
        path = self.project_dir / self.df.loc[idx, "patch_path"]
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    @staticmethod
    def _normalize_tensor(x):
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-6)

    def _augment_tensor(self, x):
        import torch
        from torchvision.transforms import functional as TF

        if self.augment:
            # Exact 90-degree rotations avoid interpolation of lattice fringes.
            k = random.randint(0, 3)
            x = torch.rot90(x, k, dims=(-2, -1))

            if random.random() < 0.5:
                x = torch.flip(x, dims=(-1,))
            if random.random() < 0.5:
                x = torch.flip(x, dims=(-2,))

            # Mild monotonic intensity changes; physical spatial scale is unchanged.
            gamma = random.uniform(0.85, 1.15)
            x = torch.clamp(x, 0.0, 1.0).pow(gamma)

            if random.random() < 0.3:
                sigma = random.uniform(0.005, 0.025)
                x = x + sigma * torch.randn_like(x)

            if random.random() < 0.2:
                ksize = random.choice([3, 5])
                x = TF.gaussian_blur(x, kernel_size=[ksize, ksize],
                                     sigma=[0.2, 1.0])

        x = torch.clamp(x, 0.0, 1.0)
        return self._normalize_tensor(x)

    def __getitem__(self, idx: int):
        x = self._load_tensor(idx)
        if self.pair_views:
            return self._augment_tensor(x.clone()), self._augment_tensor(x.clone())
        return self._normalize_tensor(x), self.df.loc[idx, "patch_id"]


def nt_xent_loss(z1, z2, temperature: float = 0.2):
    import torch
    import torch.nn.functional as F

    # Keep the encoder/projector under AMP, but compute the contrastive
    # similarity matrix in float32. This avoids fp16 overflow/underflow in
    # the diagonal mask and improves NT-Xent numerical stability.
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    z = torch.cat([z1, z2], dim=0)
    n = z1.shape[0]

    logits = (z @ z.T) / temperature
    logits.fill_diagonal_(torch.finfo(logits.dtype).min)

    targets = torch.cat(
        [
            torch.arange(n, 2 * n, device=z.device),
            torch.arange(0, n, device=z.device),
        ],
        dim=0,
    )
    return F.cross_entropy(logits, targets)


def grouped_split(
    df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    sources = sorted(df["source_id"].unique().tolist())
    if len(sources) < 3:
        raise ValueError(
            f"Need at least 3 independent source files; found {len(sources)}."
        )

    rng = random.Random(seed)
    rng.shuffle(sources)
    n = len(sources)

    if n <= 4:
        n_val = 1
        n_test = 1
    else:
        n_val = max(1, int(round(n * 0.15)))
        n_test = max(1, int(round(n * 0.15)))

    test_sources = set(sources[:n_test])
    val_sources = set(sources[n_test:n_test + n_val])
    train_sources = set(sources[n_test + n_val:])

    out = df.copy()
    out["split"] = out["source_id"].map(
        lambda s: (
            "test" if s in test_sources
            else "val" if s in val_sources
            else "train"
        )
    )
    return out


def balance_by_source(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Subsample every source down to the smallest source's patch count.

    Sources contribute wildly different patch counts (a low-magnification
    image yields far fewer physical-FOV crops than a high-magnification one),
    so without this a single source can dominate the contrastive batches.
    """
    counts = df.groupby("source_id").size()
    if counts.empty or counts.nunique() == 1:
        return df.reset_index(drop=True)

    n_keep = int(counts.min())
    rng = np.random.default_rng(seed)
    parts = []
    for _, part in df.groupby("source_id", sort=True):
        idx = np.sort(rng.choice(len(part), size=n_keep, replace=False))
        parts.append(part.iloc[idx])
    return pd.concat(parts).reset_index(drop=True)


def detect_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_group(
    project: Path,
    manifest: pd.DataFrame,
    group: str,
    args: argparse.Namespace,
    seed: int,
) -> None:
    import torch
    from torch.utils.data import DataLoader

    group_df = manifest[manifest["scale_group"] == group].copy()
    n_sources = group_df["source_id"].nunique()
    if n_sources < 3:
        print(
            f"SKIP {group}: only {n_sources} independent source file(s); "
            "need >=3 for train/val/test source-level split."
        )
        return

    split_df = grouped_split(group_df, seed)
    model_dir = project / "models" / group
    model_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(model_dir / "split_manifest.csv", index=False)

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]

    if getattr(args, "balance_sources", False):
        before = len(train_df)
        train_df = balance_by_source(train_df, seed)
        print(
            f"[{group}] source-balanced training set: "
            f"{before} -> {len(train_df)} patches "
            f"({train_df.source_id.nunique()} sources x "
            f"{len(train_df) // max(1, train_df.source_id.nunique())})"
        )

    print(
        f"\n[{group}] sources train/val/test = "
        f"{train_df.source_id.nunique()}/"
        f"{val_df.source_id.nunique()}/"
        f"{test_df.source_id.nunique()}"
    )
    print(
        f"[{group}] patches train/val/test = "
        f"{len(train_df)}/{len(val_df)}/{len(test_df)}"
    )

    train_ds = PatchDataset(train_df, project, pair_views=True, augment=True)
    val_ds = PatchDataset(val_df, project, pair_views=True, augment=True)

    batch_size = min(args.batch_size, max(2, len(train_ds)))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=(len(train_ds) > batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(args.batch_size, max(2, len(val_ds))),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    device = detect_device(args.device)
    model = build_model(args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for x1, x2 in train_loader:
            if x1.shape[0] < 2:
                continue
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = nt_xent_loss(z1, z2, args.temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x1, x2 in val_loader:
                if x1.shape[0] < 2:
                    continue
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    _, z1 = model(x1)
                    _, z2 = model(x2)
                    loss = nt_xent_loss(z1, z2, args.temperature)
                val_losses.append(float(loss.detach().cpu()))

        scheduler.step()

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"[{group}] epoch {epoch:03d}/{args.epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f}"
        )

        state = {
            "model_state": model.state_dict(),
            "group": group,
            "projection_dim": args.projection_dim,
            "embedding_dim": model.embedding_dim,
            "epoch": epoch,
            "seed": seed,
        }
        torch.save(state, model_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, model_dir / "best.pt")

    pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)
    print(f"[{group}] best validation loss: {best_val:.4f}")


def command_train(args: argparse.Namespace) -> None:
    project = args.project_dir.resolve()
    manifest = pd.read_csv(project / "dataset" / "manifest.csv")
    config = load_config(project / "pilot_config_used.json")
    seed = int(config.get("seed", 42))
    set_seed(seed)

    available = sorted(manifest["scale_group"].unique())
    groups = available if args.all_eligible else args.groups

    if not groups:
        raise RuntimeError(
            f"No group specified. Available groups: {', '.join(available)}"
        )

    for group in groups:
        if group not in available:
            print(f"SKIP unknown group: {group}")
            continue
        train_one_group(project, manifest, group, args, seed)


# ---------------------------------------------------------------------
# Handcrafted features
# ---------------------------------------------------------------------
def handcrafted_features(
    arr01: np.ndarray,
    patch_fov_nm: float | None = None,
    patch_fov_nm_y: float | None = None,
    source_pixels: int | None = None,
) -> dict[str, float]:
    from scipy import stats
    from skimage import feature, filters
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    img = np.asarray(arr01, dtype=np.float32)
    v = img.ravel()

    q05, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
    hist, _ = np.histogram(v, bins=64, range=(0, 1))
    p = hist.astype(float)
    p /= max(p.sum(), 1.0)
    pnz = p[p > 0]
    entropy = float(-np.sum(pnz * np.log2(pnz)))

    out = {
        "int_mean": float(v.mean()),
        "int_std": float(v.std()),
        "int_q05": float(q05),
        "int_q25": float(q25),
        "int_median": float(q50),
        "int_q75": float(q75),
        "int_q95": float(q95),
        "int_iqr": float(q75 - q25),
        "int_skew": float(stats.skew(v, bias=False)) if v.std() > 0 else 0.0,
        "int_kurtosis": float(stats.kurtosis(v, fisher=True, bias=False))
        if v.std() > 0 else 0.0,
        "int_entropy": entropy,
    }

    # GLCM: after physical-FOV normalization, distances in output pixels are
    # comparable within a scale group.
    levels = 32
    q = np.clip(np.floor(img * levels), 0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(
        q,
        distances=[1, 2, 4, 8],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    for prop in [
        "contrast", "dissimilarity", "homogeneity",
        "ASM", "energy", "correlation"
    ]:
        vals = graycoprops(glcm, prop)
        key = prop.lower()
        out[f"glcm_{key}_mean"] = float(np.nanmean(vals))
        out[f"glcm_{key}_std"] = float(np.nanstd(vals))

    u8 = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(u8, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(11), range=(0, 10))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= max(lbp_hist.sum(), 1.0)
    for i, val in enumerate(lbp_hist):
        out[f"lbp8r1_bin{i:02d}"] = float(val)

    grad = filters.sobel(img)
    canny = feature.canny(img, sigma=1.0)
    out["edge_sobel_mean"] = float(grad.mean())
    out["edge_sobel_std"] = float(grad.std())
    out["edge_canny_fraction"] = float(canny.mean())

    x = img - img.mean()
    F = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(F) ** 2
    h, w = img.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    FX, FY = np.meshgrid(fx, fy)
    r = np.sqrt(FX ** 2 + FY ** 2)
    mask = r > 0
    rr, pp = r[mask], power[mask]
    total = float(pp.sum())

    if total > 0:
        out["fft_low_frac"] = float(pp[rr < 0.10].sum() / total)
        out["fft_mid_frac"] = float(
            pp[(rr >= 0.10) & (rr < 0.25)].sum() / total
        )
        out["fft_high_frac"] = float(pp[rr >= 0.25].sum() / total)
        out["fft_centroid"] = float((rr * pp).sum() / total)

        fxv, fyv = FX[mask], FY[mask]
        mxx = float((pp * fxv * fxv).sum() / total)
        myy = float((pp * fyv * fyv).sum() / total)
        mxy = float((pp * fxv * fyv).sum() / total)
        ev = np.linalg.eigvalsh(np.array([[mxx, mxy], [mxy, myy]]))
        ev = np.maximum(ev, 0)
        out["fft_anisotropy"] = float(
            (ev[-1] - ev[0]) / (ev[-1] + ev[0] + 1e-12)
        )
    else:
        for k in [
            "fft_low_frac", "fft_mid_frac", "fft_high_frac",
            "fft_centroid", "fft_anisotropy"
        ]:
            out[k] = 0.0

    # fft_anisotropy above is a second-moment measure: it reports how elongated
    # the whole power distribution is, so astigmatism and a genuine pair of
    # Bragg reflections look alike. The azimuthal max/median ratio instead
    # detects discrete peaks against the diffuse halo at the same radius.
    if patch_fov_nm is not None and np.isfinite(patch_fov_nm):
        bragg = azimuthal_bragg_score(
            img,
            float(patch_fov_nm),
            patch_fov_nm_y=patch_fov_nm_y,
            source_pixels=source_pixels,
        )
        out["fft_bragg_log_ratio"] = float(
            np.log10(max(bragg["bragg_ratio"], 1.0))
        )
        out["fft_bragg_pixels"] = float(bragg["bragg_pixels"])

    return out


def extract_group_features(
    project: Path,
    manifest: pd.DataFrame,
    group: str,
    args: argparse.Namespace,
) -> None:
    import torch
    from torch.utils.data import DataLoader

    model_path = project / "models" / group / "best.pt"
    if not model_path.exists():
        print(f"SKIP {group}: no checkpoint {model_path}")
        return

    group_df = manifest[manifest["scale_group"] == group].copy().reset_index(drop=True)
    if group_df.empty:
        return

    device = detect_device(args.device)
    state = torch.load(model_path, map_location=device)
    model = build_model(int(state.get("projection_dim", 128))).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    ds = PatchDataset(group_df, project, pair_views=False, augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    embeddings = []
    patch_ids = []

    with torch.no_grad():
        for x, ids in loader:
            x = x.to(device, non_blocking=True)
            h, _ = model(x)
            embeddings.append(h.cpu().numpy())
            patch_ids.extend(list(ids))

    emb = np.concatenate(embeddings, axis=0)
    emb_cols = [f"ssl_{i:04d}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols)
    emb_df.insert(0, "patch_id", patch_ids)

    base_cols = [
        "patch_id", "patch_path", "source_id", "source_file",
        "scale_group", "patch_fov_nm", "x_nm_source", "y_nm_source"
    ]
    out_emb = group_df[base_cols].merge(emb_df, on="patch_id", how="left")

    feat_dir = project / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_emb.to_csv(feat_dir / f"ssl_embeddings_{group}.csv", index=False)

    handcrafted_rows = []
    for i, row in group_df.iterrows():
        path = project / row["patch_path"]
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        # Prefer the calibrated per-axis field of view; older manifests only
        # carry the nominal square value.
        fov_x = row.get("patch_fov_nm_x")
        if fov_x is None or not np.isfinite(fov_x):
            fov_x = row.get("patch_fov_nm")
        f = handcrafted_features(
            arr,
            fov_x,
            patch_fov_nm_y=row.get("patch_fov_nm_y"),
            source_pixels=row.get("crop_size_px_source"),
        )
        handcrafted_rows.append({
            **{c: row[c] for c in base_cols},
            **f,
        })

    pd.DataFrame(handcrafted_rows).to_csv(
        feat_dir / f"handcrafted_{group}.csv",
        index=False,
    )
    print(
        f"[{group}] extracted {emb.shape[1]}-D SSL embeddings "
        f"and handcrafted features for {len(group_df)} patches."
    )


def command_extract(args: argparse.Namespace) -> None:
    project = args.project_dir.resolve()
    manifest = pd.read_csv(project / "dataset" / "manifest.csv")
    available = sorted(manifest["scale_group"].unique())
    groups = available if args.all_trained else args.groups

    if not groups:
        raise RuntimeError(
            f"No group specified. Available groups: {', '.join(available)}"
        )

    for group in groups:
        extract_group_features(project, manifest, group, args)


# ---------------------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------------------
def prune_correlated_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    corr_threshold: float,
):
    X = df[feature_cols].copy()
    report = []

    # Remove non-finite / constant features.
    kept = []
    for c in feature_cols:
        v = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
        if not np.isfinite(v).all():
            report.append({
                "feature": c, "selected": False,
                "reason": "non_finite", "paired_with": "",
                "abs_correlation": np.nan,
            })
            continue
        if np.max(v) - np.min(v) <= 1e-12 * max(1.0, np.max(np.abs(v))):
            report.append({
                "feature": c, "selected": False,
                "reason": "constant_or_near_constant", "paired_with": "",
                "abs_correlation": np.nan,
            })
            continue
        kept.append(c)

    X = X[kept]
    # X.corr() on a frame with no columns returns an empty frame; guard anyway
    # so the caller always receives a well-formed (possibly empty) matrix.
    corr = X.corr().abs() if kept else pd.DataFrame()

    dropped = {}
    cols = list(X.columns)
    for j in range(1, len(cols)):
        cj = cols[j]
        if cj in dropped:
            continue
        for i in range(j):
            ci = cols[i]
            if ci in dropped:
                continue
            r = corr.loc[ci, cj]
            if np.isfinite(r) and r >= corr_threshold:
                dropped[cj] = (ci, float(r))
                break

    selected = [c for c in cols if c not in dropped]

    for c, (paired, r) in dropped.items():
        report.append({
            "feature": c, "selected": False,
            "reason": f"high_correlation_ge_{corr_threshold:.3f}",
            "paired_with": paired, "abs_correlation": r,
        })
    for c in selected:
        report.append({
            "feature": c, "selected": True,
            "reason": "retained", "paired_with": "",
            "abs_correlation": np.nan,
        })

    return selected, pd.DataFrame(report), corr


def pca_and_cluster(
    df: pd.DataFrame,
    feature_cols: list[str],
    metadata_cols: list[str],
    standardize: bool,
):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler, normalize

    X = df[feature_cols].to_numpy(float)
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(
            f"PCA needs at least one sample and one feature; got {X.shape}."
        )

    if standardize:
        X = StandardScaler().fit_transform(X)
    else:
        X = normalize(X, norm="l2")

    # PCA requires n_components <= min(n_samples, n_features); the old
    # max(2, ...) floor raised that ceiling and crashed on small or
    # nearly-degenerate feature sets. Fewer than two components simply means
    # no PC2 column downstream, which the viewer already tolerates.
    n_components = min(20, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)

    out = df[metadata_cols].copy()
    for i in range(Z.shape[1]):
        out[f"PC{i+1}"] = Z[:, i]

    # Simple exploratory KMeans only; choose k by silhouette.
    best_k = None
    best_score = -np.inf
    if len(df) >= 6:
        max_k = min(6, len(df) - 1)
        for k in range(2, max_k + 1):
            labels = KMeans(
                n_clusters=k,
                random_state=42,
                n_init="auto",
            ).fit_predict(Z[:, :min(10, Z.shape[1])])
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(
                Z[:, :min(10, Z.shape[1])],
                labels,
            )
            if score > best_score:
                best_score = score
                best_k = k

    if best_k is not None:
        labels = KMeans(
            n_clusters=best_k,
            random_state=42,
            n_init="auto",
        ).fit_predict(Z[:, :min(10, Z.shape[1])])
        out["cluster_kmeans"] = labels
        out["kmeans_k"] = best_k
        out["kmeans_silhouette"] = best_score

    return out, pca.explained_variance_ratio_


def nearest_neighbours_cross_source(
    emb_df: pd.DataFrame,
    feature_cols: list[str],
    top_k: int = 5,
    block_size: int = 512,
) -> pd.DataFrame:
    from sklearn.preprocessing import normalize

    X = normalize(emb_df[feature_cols].to_numpy(float), norm="l2")
    patch_ids = emb_df["patch_id"].to_numpy()
    source_ids = emb_df["source_id"].to_numpy()
    n = len(X)

    rows = []

    # Similarities are computed in row blocks. Materializing the full N x N
    # matrix costs 8*N^2 bytes -- already ~5 GB for 25k patches -- while only
    # top_k cross-source neighbours per row are ever used.
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sim_block = X[start:stop] @ X.T

        for local_i, i in enumerate(range(start, stop)):
            sims = sim_block[local_i]

            # Same-source patches are excluded, which also removes j == i.
            eligible = np.flatnonzero(source_ids != source_ids[i])
            if eligible.size == 0:
                continue

            k = min(top_k, eligible.size)
            # argpartition finds the top k without sorting the whole row.
            top = eligible[np.argpartition(-sims[eligible], k - 1)[:k]]
            top = top[np.argsort(-sims[top])]

            for rank, j in enumerate(top, start=1):
                rows.append({
                    "query_patch_id": patch_ids[i],
                    "query_source_id": source_ids[i],
                    "neighbor_rank": rank,
                    "neighbor_patch_id": patch_ids[j],
                    "neighbor_source_id": source_ids[j],
                    "cosine_similarity": float(sims[j]),
                })

    return pd.DataFrame(rows)


def crystallinity_probe(
    project: Path,
    group: str,
    ssl_df: pd.DataFrame,
    hand_df: pd.DataFrame,
    metadata_cols: list[str],
    hi_quantile: float,
    lo_quantile: float,
) -> pd.DataFrame | None:
    """
    Linear probe: can a representation tell crystalline from amorphous patches?

    The label is the FFT Bragg score recorded during prepare, thresholded at
    two quantiles so only confident patches are used; the ambiguous middle is
    discarded. Folds are held out by source_id, because a patch-level split
    would let the probe recognise the source image instead of the structure.

    The score is derived from the power spectrum, so handcrafted FFT features
    are partly circular with the label and their AUC is not comparable to the
    others. The SSL columns are the honest comparison: PatchDataset
    standardises every patch, so the encoder never sees absolute contrast.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler, normalize

    manifest_path = project / "dataset" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if "bragg_ratio" not in manifest.columns:
        print(
            f"[{group}] no bragg_ratio in {manifest_path.name}; "
            "re-run prepare to enable the crystallinity probe."
        )
        return None

    label_src = manifest[["patch_id", "bragg_ratio"]]
    df = ssl_df.merge(label_src, on="patch_id", how="inner")
    if df.empty or not np.isfinite(df["bragg_ratio"]).any():
        return None

    # Identify the checkpoint's held-out sources *before* touching
    # bragg_ratio at all. The encoder was fitted on the `train` sources and
    # selected on `val`, so scoring those sources would report how well the
    # probe reads patches the representation has already seen -- and fitting
    # the crystalline/amorphous quantile cut on those same held-out scores
    # would be transductive rather than a genuine held-out evaluation: the
    # test sources' own values would help decide the very threshold used to
    # label them.
    holdout: set[str] = set()
    split_path = project / "models" / group / "split_manifest.csv"
    if split_path.exists():
        try:
            split_df = pd.read_csv(split_path)
            holdout = set(
                split_df.loc[split_df["split"] == "test", "source_id"].unique()
            )
        except (ValueError, KeyError):
            holdout = set()

    # Thresholds are fit on non-held-out sources only, when there are any;
    # an empty holdout (no checkpoint split, i.e. the LOSO fallback below)
    # simply leaves the full pool, which is the only option when there is no
    # held-out source to protect.
    fit_mask = (
        ~df["source_id"].isin(holdout) if holdout
        else pd.Series(True, index=df.index)
    )
    threshold_pool = df.loc[fit_mask, "bragg_ratio"]
    if threshold_pool.empty:
        # A holdout covering every source in df leaves nothing to fit from;
        # fall back to the full pool rather than fail outright.
        threshold_pool = df["bragg_ratio"]

    hi = float(threshold_pool.quantile(hi_quantile))
    lo = float(threshold_pool.quantile(lo_quantile))
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        print(f"[{group}] bragg_ratio has no usable spread; skipping probe.")
        return None

    keep = (df["bragg_ratio"] >= hi) | (df["bragg_ratio"] <= lo)
    df = df[keep].copy()
    df["label"] = (df["bragg_ratio"] >= hi).astype(int)

    def both_classes(sources) -> list[str]:
        return [
            str(s) for s in sources
            if df.loc[df["source_id"] == s, "label"].nunique() == 2
        ]

    # The same split is applied to every representation block below, keeping
    # the reported numbers comparable across them.
    present = set(df["source_id"].unique())
    eval_sources = both_classes(sorted(holdout & present))
    if eval_sources:
        protocol = "encoder-held-out"
        folds = [
            (
                np.flatnonzero(~df["source_id"].isin(eval_sources).to_numpy()),
                np.flatnonzero((df["source_id"] == s).to_numpy()),
            )
            for s in eval_sources
        ]
    else:
        usable = both_classes(sorted(present))
        if len(usable) < 2:
            print(
                f"[{group}] fewer than 2 sources contain both classes; "
                "skipping crystallinity probe."
            )
            return None
        # No checkpoint split available (or its test source lacks both
        # classes): fall back to plain leave-one-source-out and say so, since
        # the encoder has seen most of these sources.
        protocol = "all-sources-encoder-exposed"
        eval_sources = usable
        folds = [
            (
                np.flatnonzero((df["source_id"] != s).to_numpy()),
                np.flatnonzero((df["source_id"] == s).to_numpy()),
            )
            for s in usable
        ]

    hand_cols = [c for c in hand_df.columns if c not in metadata_cols]
    hand_aligned = df[["patch_id"]].merge(
        hand_df[["patch_id"] + hand_cols], on="patch_id", how="left"
    )
    ssl_cols = [c for c in df.columns if c.startswith("ssl_")]

    # The labels come from the patch's own power spectrum, so *every* fft_*
    # descriptor shares their source, not just the fft_bragg_* pair that
    # reproduces bragg_ratio outright. Any block containing them is circular
    # to some degree, so they are reported for reference but kept out of the
    # ranking; only spectrum-independent features form the honest baseline.
    bragg_cols = [c for c in hand_cols if c.startswith("fft_bragg")]
    # Everything spectrum-derived, used to keep the ranked baseline clean...
    spectral_cols = [c for c in hand_cols if c.startswith("fft_")]
    # ...while the generic FFT reference must exclude the Bragg columns, or it
    # would carry the labelling score itself and stop measuring the remaining
    # spectral descriptors.
    fft_cols = [c for c in spectral_cols if c not in bragg_cols]
    spatial_cols = [c for c in hand_cols if c not in spectral_cols]

    # (matrix, needs_standardising, circular, caveat). L2 row normalisation is
    # per sample and cannot leak; StandardScaler is per column and must
    # therefore be fitted inside each fold, below.
    blocks = {
        "ssl_embedding": (
            normalize(df[ssl_cols].to_numpy(float), norm="l2"),
            False, False, "contrast-blind",
        ),
        "handcrafted_spatial": (
            hand_aligned[spatial_cols].to_numpy(float),
            True, False, "contrast-visible, spectrum-independent",
        ),
        "handcrafted_fft": (
            hand_aligned[fft_cols].to_numpy(float),
            True, True, "shares the label's power spectrum (not ranked)",
        ),
        "handcrafted_bragg": (
            hand_aligned[bragg_cols].to_numpy(float),
            True, True, "reproduces the label itself (not ranked)",
        ),
    }

    y = df["label"].to_numpy()
    rows = []
    for name, (X, needs_scaling, circular, caveat) in blocks.items():
        if X.shape[1] == 0:
            continue
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        aucs = []
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            x_tr, x_te = X[tr], X[te]
            if needs_scaling:
                # Fitting on the whole set would carry the held-out source's
                # mean and variance into the fold, and LogisticRegression is
                # L2-regularised by default, so that shifts the effective
                # penalty and inflates the reported cross-source AUC.
                scaler = StandardScaler().fit(x_tr)
                x_tr = scaler.transform(x_tr)
                x_te = scaler.transform(x_te)
            clf = LogisticRegression(max_iter=3000).fit(x_tr, y[tr])
            aucs.append(
                roc_auc_score(y[te], clf.predict_proba(x_te)[:, 1])
            )
        if not aucs:
            continue
        rows.append({
            "representation": name,
            "dimensions": int(X.shape[1]),
            "circular": circular,
            "caveat": caveat,
            "folds": len(aucs),
            "auc_mean": float(np.mean(aucs)),
            # A single held-out source gives no spread to report; 0.0 would
            # read as a tight, well-replicated estimate.
            "auc_std": float(np.std(aucs)) if len(aucs) > 1 else float("nan"),
            "auc_min": float(np.min(aucs)),
        })

    if not rows:
        return None

    # Circular rows sort last so they can never be read as the best result.
    out = pd.DataFrame(rows).sort_values(
        ["circular", "auc_mean"], ascending=[True, False]
    )
    # Counts must describe the rows actually scored: under encoder-held-out
    # only the test sources are evaluated, so whole-dataset totals would
    # overstate the evaluation sample by several times.
    is_eval = df["source_id"].isin(eval_sources).to_numpy()
    out["protocol"] = protocol
    out["eval_sources"] = ";".join(eval_sources)
    out["n_eval_crystalline"] = int(y[is_eval].sum())
    out["n_eval_amorphous"] = int((1 - y[is_eval]).sum())
    # Under leave-one-source-out the fit set differs per fold, so a single
    # figure would be wrong; only the fixed encoder-held-out split has one.
    out["n_fit_patches"] = (
        float((~is_eval).sum())
        if protocol == "encoder-held-out"
        else float("nan")
    )
    out["n_total_crystalline"] = int(y.sum())
    out["n_total_amorphous"] = int((1 - y).sum())
    out["threshold_hi"] = hi
    out["threshold_lo"] = lo
    return out


def write_crystallinity_probe(
    project: Path,
    group: str,
    analysis_dir: Path,
    ssl_df: pd.DataFrame,
    hand_df: pd.DataFrame,
    metadata_cols: list[str],
    hi_quantile: float,
    lo_quantile: float,
) -> None:
    """
    Run the probe and publish its result, or leave no result behind.

    The CSV is removed first: when the probe cannot run this time -- an old
    manifest, constant scores, too few sources carrying both classes, or
    shifted thresholds -- a file from an earlier run would otherwise be read
    as the current analysis.
    """
    out_path = analysis_dir / "crystallinity_probe.csv"
    out_path.unlink(missing_ok=True)

    probe = crystallinity_probe(
        project, group, ssl_df, hand_df, metadata_cols,
        hi_quantile, lo_quantile,
    )
    if probe is None:
        return

    probe.to_csv(out_path, index=False)
    ranked = probe[~probe["circular"]]
    if ranked.empty:
        return
    best = ranked.iloc[0]
    ssl_row = probe[probe["representation"] == "ssl_embedding"]
    ssl_txt = (
        f", SSL AUC={ssl_row.auc_mean.iloc[0]:.3f}" if not ssl_row.empty else ""
    )
    print(
        f"[{group}] crystallinity probe [{best.protocol}]: evaluated on "
        f"{int(best.n_eval_crystalline)} crystalline / "
        f"{int(best.n_eval_amorphous)} amorphous held-out patches over "
        f"{int(best.folds)} fold(s) "
        f"(of {int(best.n_total_crystalline)}/{int(best.n_total_amorphous)} "
        f"labelled in total); best={best.representation} "
        f"AUC={best.auc_mean:.3f}{ssl_txt}"
    )


def analyze_group(
    project: Path,
    group: str,
    corr_threshold: float,
    probe_hi_quantile: float = 0.85,
    probe_lo_quantile: float = 0.35,
) -> None:
    feat_dir = project / "features"
    analysis_dir = project / "analysis" / group
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ssl_path = feat_dir / f"ssl_embeddings_{group}.csv"
    hand_path = feat_dir / f"handcrafted_{group}.csv"

    if not ssl_path.exists() or not hand_path.exists():
        print(f"SKIP {group}: feature files are missing; run extract first.")
        return

    metadata_cols = [
        "patch_id", "patch_path", "source_id", "source_file",
        "scale_group", "patch_fov_nm", "x_nm_source", "y_nm_source"
    ]

    ssl = pd.read_csv(ssl_path)
    ssl_cols = [c for c in ssl.columns if c.startswith("ssl_")]
    ssl_pca, ssl_var = pca_and_cluster(
        ssl, ssl_cols, metadata_cols,
        standardize=False,
    )
    ssl_pca.to_csv(analysis_dir / "pca_ssl.csv", index=False)
    pd.DataFrame({
        "component": np.arange(1, len(ssl_var) + 1),
        "explained_variance_ratio": ssl_var,
        "cumulative_variance": np.cumsum(ssl_var),
    }).to_csv(analysis_dir / "pca_ssl_variance.csv", index=False)

    nearest_neighbours_cross_source(
        ssl,
        ssl_cols,
        top_k=5,
    ).to_csv(analysis_dir / "cross_source_nearest_neighbors.csv", index=False)

    hand = pd.read_csv(hand_path)
    hand_cols = [c for c in hand.columns if c not in metadata_cols]
    selected, report, corr = prune_correlated_features(
        hand,
        hand_cols,
        corr_threshold,
    )
    report.to_csv(analysis_dir / "handcrafted_feature_selection.csv", index=False)
    corr.to_csv(analysis_dir / "handcrafted_feature_correlation.csv")

    selected_raw = hand[metadata_cols + selected].copy()
    selected_raw.to_csv(analysis_dir / "handcrafted_selected_raw.csv", index=False)

    # fft_bragg_* can survive correlation pruning (it usually does: nothing
    # else in the handcrafted set is a close proxy for it) and would then
    # enter pca_handcrafted.csv. The viewer colours that PCA by Bragg-score
    # band, so a feature computed from bragg_ratio itself would make any
    # separation tautological rather than the honest SSL-vs-handcrafted
    # comparison this PCA is meant to be. Excluded here, not upstream in
    # `selected`, so handcrafted_selected_raw.csv above still records the
    # full selection for reference.
    pca_cols = [c for c in selected if not c.startswith("fft_bragg")]

    if not pca_cols:
        # A previous run may have written PCA outputs that no longer match the
        # current feature files; drop them rather than let the viewer show
        # results that were never produced from this analysis.
        for stale in ("pca_handcrafted.csv", "pca_handcrafted_variance.csv"):
            (analysis_dir / stale).unlink(missing_ok=True)
        print(
            f"[{group}] no handcrafted feature survived selection "
            f"({len(hand_cols)} candidates were non-finite, constant, "
            "redundant, or Bragg-derived); skipping handcrafted PCA."
        )
        write_crystallinity_probe(
            project, group, analysis_dir, ssl, hand, metadata_cols,
            probe_hi_quantile, probe_lo_quantile,
        )
        return

    hand_pca, hand_var = pca_and_cluster(
        selected_raw,
        pca_cols,
        metadata_cols,
        standardize=True,
    )
    hand_pca.to_csv(analysis_dir / "pca_handcrafted.csv", index=False)
    pd.DataFrame({
        "component": np.arange(1, len(hand_var) + 1),
        "explained_variance_ratio": hand_var,
        "cumulative_variance": np.cumsum(hand_var),
    }).to_csv(analysis_dir / "pca_handcrafted_variance.csv", index=False)

    write_crystallinity_probe(
        project, group, analysis_dir, ssl, hand, metadata_cols,
        probe_hi_quantile, probe_lo_quantile,
    )

    print(
        f"[{group}] analysis complete: "
        f"handcrafted {len(hand_cols)} -> {len(selected)} selected."
    )


def command_analyze(args: argparse.Namespace) -> None:
    project = args.project_dir.resolve()
    feat_dir = project / "features"
    groups = sorted(
        p.stem.replace("ssl_embeddings_", "")
        for p in feat_dir.glob("ssl_embeddings_*.csv")
    )
    requested = groups if args.all_extracted else args.groups

    if not requested:
        raise RuntimeError(
            f"No group specified. Extracted groups: {', '.join(groups)}"
        )

    for group in requested:
        analyze_group(
            project, group, args.corr_threshold,
            args.probe_hi_quantile, args.probe_lo_quantile,
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Physical-scale-aware TEM self-supervised-learning pilot pipeline"
    )
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Create physical-scale-aware patch dataset")
    prep.add_argument("input_dir", type=Path)
    prep.add_argument("project_dir", type=Path)
    prep.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("pilot_config.json"),
    )
    prep.add_argument(
        "--fallback-nm-per-pixel",
        type=float,
        default=None,
        help="Only for raster images or files whose physical scale is missing.",
    )
    prep.set_defaults(func=command_prepare)

    train = sub.add_parser("train", help="Train SimCLR model(s)")
    train.add_argument("project_dir", type=Path)
    train.add_argument("--groups", nargs="*", default=[])
    train.add_argument("--all-eligible", action="store_true")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--temperature", type=float, default=0.2)
    train.add_argument("--projection-dim", type=int, default=128)
    train.add_argument(
        "--balance-sources",
        action="store_true",
        help="Subsample each training source to the smallest source's patch "
             "count so one image cannot dominate the contrastive batches.",
    )
    train.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cpu, mps, cuda:0, ...",
    )
    train.set_defaults(func=command_train)

    extract = sub.add_parser("extract", help="Extract SSL and handcrafted features")
    extract.add_argument("project_dir", type=Path)
    extract.add_argument("--groups", nargs="*", default=[])
    extract.add_argument("--all-trained", action="store_true")
    extract.add_argument("--batch-size", type=int, default=128)
    extract.add_argument("--workers", type=int, default=4)
    extract.add_argument("--device", default="auto")
    extract.set_defaults(func=command_extract)

    analyze = sub.add_parser("analyze", help="Feature selection + PCA + clustering")
    analyze.add_argument("project_dir", type=Path)
    analyze.add_argument("--groups", nargs="*", default=[])
    analyze.add_argument("--all-extracted", action="store_true")
    analyze.add_argument("--corr-threshold", type=float, default=0.95)
    analyze.add_argument(
        "--probe-hi-quantile",
        type=float,
        default=0.85,
        help="Patches at or above this bragg_ratio quantile are labelled "
             "crystalline for the linear probe.",
    )
    analyze.add_argument(
        "--probe-lo-quantile",
        type=float,
        default=0.35,
        help="Patches at or below this bragg_ratio quantile are labelled "
             "amorphous; the ambiguous middle is discarded.",
    )
    analyze.set_defaults(func=command_analyze)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
