#!/usr/bin/env python3
"""
Pure helpers for the viewer.

Kept out of ``viewer.py`` because that module is a Streamlit script: importing
it executes the page. Everything here is side-effect free and importable from
tests without Streamlit.

Nothing in here writes to the dataset. The Bragg *bands* are a viewer-side
reading of a continuous score, never a stored label and never a phase
assignment -- see ``BRAGG_CAVEAT``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Scale groups
# ---------------------------------------------------------------------
# `scale_group` records the physical sampling scale a patch was cropped at.
# It says nothing about crystallinity or material class -- an amorphous region
# imaged at 0.007 nm/pixel is still "lattice". The internal keys are part of
# the on-disk layout (directory names, CSV values) and must not change, so the
# rewording lives here, in the display layer only.
SCALE_GROUP_LABELS: dict[str, str] = {
    "lattice": "High-resolution / atomic scale (lattice)",
    "nano": "Nanoscale morphology (nano)",
    "meso": "Aggregate / mesoscale morphology (meso)",
    "micro": "Low-magnification / micrometre scale (micro)",
}

SCALE_GROUP_NOTE = (
    "scale_group は観察スケール（物理ピクセルサイズ）の区分です。"
    "結晶性や材質のクラスではありません —— 同じ lattice 群にも"
    "格子縞の見えるパッチと非晶質のパッチが混在します。"
)


def scale_group_label(group: Any) -> str:
    """Human-readable name for a scale-group key; unknown keys pass through."""
    key = "" if group is None else str(group)
    if key in SCALE_GROUP_LABELS:
        return SCALE_GROUP_LABELS[key]
    return f"Observation scale ({key})" if key else "Observation scale"


# ---------------------------------------------------------------------
# Bragg score bands
# ---------------------------------------------------------------------
BRAGG_COLUMNS: tuple[str, ...] = ("bragg_ratio", "bragg_d_nm", "bragg_pixels")
BAND_COLUMN = "bragg_band"

BAND_HIGH = "High Bragg score"
BAND_AMBIGUOUS = "Ambiguous"
BAND_LOW = "Low Bragg score"
BAND_ORDER: tuple[str, ...] = (BAND_HIGH, BAND_AMBIGUOUS, BAND_LOW)
BAND_UNSCORED = "スコアなし"

BRAGG_CAVEAT = (
    "Bragg score はパッチ自身の FFT から計算した**相対的なスコア**です。"
    "校正済みの相ラベルではありません —— High/Low は結晶質・非晶質の断定ではなく、"
    "同一データセット内での相対的な強弱を表します。"
)

_MISSING_PROBE_NOTE = (
    "結晶性スコア解析が未実行です。"
    "`analyze` を実行すると analysis/<group>/crystallinity_probe.csv が作られ、"
    "Bragg score の帯域表示が有効になります。"
)


def probe_missing_note() -> str:
    return _MISSING_PROBE_NOTE


def read_bragg_thresholds(path: str | Path) -> tuple[float, float] | None:
    """
    Read (threshold_hi, threshold_lo) from a crystallinity_probe.csv.

    Returns None whenever the thresholds cannot be trusted -- the file is
    missing, empty, unparseable, lacks the columns, or holds non-finite or
    inverted values. Callers then fall back to showing the raw score with no
    bands, which is the conservative reading.
    """
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size == 0:
            return None
        df = pd.read_csv(p)
    except Exception:
        return None

    if df is None or df.empty:
        return None
    if "threshold_hi" not in df.columns or "threshold_lo" not in df.columns:
        return None

    def first_finite(col: str) -> float | None:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[np.isfinite(s.to_numpy(dtype="float64", na_value=np.nan))]
        return float(s.iloc[0]) if len(s) else None

    hi = first_finite("threshold_hi")
    lo = first_finite("threshold_lo")
    if hi is None or lo is None or hi <= lo:
        return None
    return hi, lo


def classify_bragg_band(
    values: Iterable[Any],
    hi: Any,
    lo: Any,
) -> pd.Series:
    """
    Label each score High / Ambiguous / Low.

    Non-numeric and non-finite entries get <NA>: an unscorable patch is left
    unbanded rather than pushed into a band it does not belong to. Invalid
    thresholds yield an all-<NA> result instead of raising.
    """
    s = values if isinstance(values, pd.Series) else pd.Series(list(values))
    numeric = pd.to_numeric(s, errors="coerce").astype("float64")
    band = pd.Series(pd.NA, index=numeric.index, dtype="string")
    if numeric.empty:
        return band

    try:
        hi_f, lo_f = float(hi), float(lo)
    except (TypeError, ValueError):
        return band
    if not (math.isfinite(hi_f) and math.isfinite(lo_f)) or hi_f <= lo_f:
        return band

    v = numeric.to_numpy(dtype="float64", na_value=np.nan)
    finite = np.isfinite(v)
    band[finite & (v >= hi_f)] = BAND_HIGH
    band[finite & (v <= lo_f)] = BAND_LOW
    band[finite & (v > lo_f) & (v < hi_f)] = BAND_AMBIGUOUS
    return band


def has_bragg_scores(df: pd.DataFrame | None) -> bool:
    """True when a frame carries at least one finite bragg_ratio."""
    if df is None or "bragg_ratio" not in getattr(df, "columns", []):
        return False
    v = pd.to_numeric(df["bragg_ratio"], errors="coerce")
    return bool(np.isfinite(v.to_numpy(dtype="float64", na_value=np.nan)).any())


def attach_bragg_columns(
    df: pd.DataFrame | None,
    manifest: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Left-join the manifest's Bragg columns onto a frame keyed by patch_id.

    Used to give the PCA tables the same score the Patches tab shows. Returns
    the input unchanged when either side lacks patch_id or the manifest has no
    Bragg columns, so an old dataset simply carries on without them.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "patch_id" not in df.columns:
        return df
    if manifest is None or not isinstance(manifest, pd.DataFrame):
        return df
    if "patch_id" not in getattr(manifest, "columns", []):
        return df

    cols = [c for c in BRAGG_COLUMNS if c in manifest.columns]
    if not cols:
        return df

    right = manifest[["patch_id", *cols]].drop_duplicates(subset="patch_id")
    left = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
    return left.merge(right, on="patch_id", how="left")


def band_summary(
    df: pd.DataFrame | None,
    band_col: str = BAND_COLUMN,
    source_col: str = "source_id",
) -> pd.DataFrame:
    """Patch and source counts per band, always in a fixed band order."""
    cols = ["band", "patches", "sources"]
    if df is None or not isinstance(df, pd.DataFrame) or band_col not in getattr(df, "columns", []):
        return pd.DataFrame(columns=cols)

    rows = []
    for name in BAND_ORDER:
        sub = df[df[band_col] == name]
        rows.append({
            "band": name,
            "patches": int(len(sub)),
            "sources": int(sub[source_col].nunique()) if source_col in sub.columns else 0,
        })
    unscored = df[df[band_col].isna()]
    if len(unscored):
        rows.append({
            "band": BAND_UNSCORED,
            "patches": int(len(unscored)),
            "sources": int(unscored[source_col].nunique()) if source_col in unscored.columns else 0,
        })
    return pd.DataFrame(rows, columns=cols)


def band_by_source(
    df: pd.DataFrame | None,
    band_col: str = BAND_COLUMN,
    source_col: str = "source_id",
) -> pd.DataFrame:
    """
    Per-source band composition.

    This is the table to read before touching the training sampler: it shows
    whether one source supplies most of one band.
    """
    if (df is None or not isinstance(df, pd.DataFrame)
            or band_col not in getattr(df, "columns", [])
            or source_col not in getattr(df, "columns", [])):
        return pd.DataFrame()
    out = (
        df.assign(**{band_col: df[band_col].fillna(BAND_UNSCORED)})
          .pivot_table(index=source_col, columns=band_col, values="patch_id",
                       aggfunc="count", fill_value=0)
          if "patch_id" in df.columns else pd.DataFrame()
    )
    if out.empty:
        return out
    ordered = [c for c in (*BAND_ORDER, BAND_UNSCORED) if c in out.columns]
    out = out[ordered]
    out.columns.name = None
    return out.reset_index()


def _distinguishing_decimals(values: np.ndarray, start: int = 4, cap: int = 15) -> int | None:
    """
    Fewest decimal places (at least `start`, for readability) that keep every
    value in `values` distinct, finite, and sign-preserving after rounding,
    or None when no count up to `cap` does. Histogram bin centers are already
    sorted, unique, and finite before rounding, so this only widens the
    precision when a fixed decimal count would otherwise merge, zero out, or
    overflow one of them.

    Distinctness alone is not enough: rounding a tiny positive center like
    5.6e-9 to 4 decimals gives exactly 0.0, which is both a different value
    from a nonzero neighbour (so the old uniqueness-only check accepted it)
    and a wrong one -- a positive score has no business becoming zero. The
    same reasoning bars rounding a finite value to +/-inf, which np.round
    will not itself produce but a caller-supplied cap could invite if this
    function is reused for values near float64's range limits.

    Returning `cap` on failure used to still round -- and could still merge
    bins nothing forced together, since values close enough that rounding
    can't separate them (spacing near float64 epsilon) are usually already
    distinct unrounded. None tells the caller to skip rounding rather than
    apply a precision known not to work.

    A single value is not exempt from the finite/nonzero check just because
    it cannot collide with a neighbour: an early return here for n <= 1 used
    to skip the loop entirely, so bragg_histogram([1e-10], bins=1) rounded
    its only center straight to 0.0 with no check that ever caught it. It now
    runs through the same loop as every other length.
    """
    n = len(values)
    nonzero = values != 0
    for d in range(start, cap + 1):
        rounded = np.round(values, d)
        if len(np.unique(rounded)) != n:
            continue
        if not np.all(np.isfinite(rounded)):
            continue
        if np.any(nonzero & (rounded == 0)):
            continue
        return d
    return None


def bragg_histogram(
    values: Iterable[Any],
    bins: int = 40,
    log: bool = True,
) -> pd.DataFrame:
    """
    Binned distribution of the score, ready for a bar chart.

    The score is heavily right-skewed (an isotropic halo sits near 1 while a
    strong reflection reaches the hundreds), so the default bins on log10.
    Returns an empty frame -- never raises -- when there is nothing to bin, and
    widens a zero-width range so constant data still produces one bar.
    """
    cols = ["bragg_ratio", "patches"]
    s = values if isinstance(values, pd.Series) else pd.Series(list(values))
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if v.size == 0:
        return pd.DataFrame(columns=cols)

    x = np.log10(v) if log else v
    lo, hi = float(x.min()), float(x.max())
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return pd.DataFrame(columns=cols)
    if hi <= lo:
        # Constant input: np.histogram rejects a zero-width range.
        lo, hi = lo - 0.5, hi + 0.5
    n_bins = max(1, int(bins))

    counts, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    labels = np.power(10.0, centers) if log else centers
    # st.bar_chart renders the x column as categorical text, so an unrounded
    # float prints 15+ digits per tick and the axis becomes unreadable. A
    # fixed 4 decimals collapsed distinct bins when the whole range is
    # narrower than that (e.g. scores in [1.0, 1.00001]), so the precision is
    # chosen per call: the fewest decimals -- starting from a readable 4 --
    # that still keeps every bin center distinct.
    decimals = _distinguishing_decimals(labels, start=4)
    if decimals is not None:
        labels = np.round(labels, decimals)
    # else: leave the raw floats -- an unreadable tick beats a merged bin.
    return pd.DataFrame({cols[0]: labels, cols[1]: counts})
