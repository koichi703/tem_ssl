"""Tests for tem_ssl_pilot.crystallinity_probe's threshold-fitting logic."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tem_ssl_pilot as T


def _project(tmp_path, manifest, split_rows=None):
    project = tmp_path
    (project / "dataset").mkdir()
    manifest.to_csv(project / "dataset" / "manifest.csv", index=False)
    if split_rows is not None:
        (project / "models" / "g").mkdir(parents=True)
        pd.DataFrame(split_rows).to_csv(
            project / "models" / "g" / "split_manifest.csv", index=False
        )
    return project


def _feature_frames(manifest, seed=0):
    rng = np.random.default_rng(seed)
    n = len(manifest)
    ssl_cols = {f"ssl_{i:04d}": rng.normal(size=n) for i in range(8)}
    ssl_df = pd.DataFrame({
        "patch_id": manifest.patch_id, "source_id": manifest.source_id, **ssl_cols,
    })
    hand_df = pd.DataFrame({
        "patch_id": manifest.patch_id, "source_id": manifest.source_id,
        "handcrafted_x": rng.normal(size=n),
    })
    return ssl_df, hand_df


METADATA_COLS = [
    "patch_id", "patch_path", "source_id", "source_file",
    "scale_group", "patch_fov_nm", "x_nm_source", "y_nm_source",
]


def _manifest_row(source_id, patch_id, bragg_ratio):
    return {
        "patch_id": patch_id, "source_id": source_id, "bragg_ratio": bragg_ratio,
        "patch_path": "x", "source_file": "x", "scale_group": "g",
        "patch_fov_nm": 4.0, "x_nm_source": 0.0, "y_nm_source": 0.0,
    }


def test_thresholds_stay_non_held_out_on_loso_fallback(tmp_path):
    # A checkpoint split exists, but the designated test source's scores are
    # too narrow to carry both classes under a held-out-only threshold, so
    # the probe must fall back to leave-one-source-out. The thresholds used
    # for that fallback must be the SAME non-held-out-only ones already
    # computed for the encoder-held-out attempt -- not refit from the full
    # pool (which would let the held-out source's own scores influence which
    # exposed patches get labeled, even though the source itself is later
    # dropped from evaluation) and not the full-pool quantiles either.
    rng = np.random.default_rng(0)
    rows = []
    for src in ("s1", "s2", "s3"):
        for i in range(40):
            rows.append(_manifest_row(src, f"{src}_p{i}", float(rng.uniform(1, 200))))
    for i in range(20):
        rows.append(_manifest_row("s_test", f"s_test_p{i}", 1.5 + float(rng.uniform(-0.01, 0.01))))
    manifest = pd.DataFrame(rows)
    ssl_df, hand_df = _feature_frames(manifest)

    project = _project(tmp_path, manifest, split_rows={
        "source_id": ["s1", "s2", "s3", "s_test"],
        "split": ["train", "train", "val", "test"],
    })

    probe = T.crystallinity_probe(project, "g", ssl_df, hand_df, METADATA_COLS, 0.85, 0.35)
    assert probe is not None
    row = probe.iloc[0]
    assert row["protocol"] == "all-sources-encoder-exposed"

    full_hi = manifest["bragg_ratio"].quantile(0.85)
    full_lo = manifest["bragg_ratio"].quantile(0.35)
    non_held_out_hi = manifest.loc[manifest.source_id != "s_test", "bragg_ratio"].quantile(0.85)
    non_held_out_lo = manifest.loc[manifest.source_id != "s_test", "bragg_ratio"].quantile(0.35)

    assert row["threshold_hi"] == pytest.approx(non_held_out_hi)
    assert row["threshold_lo"] == pytest.approx(non_held_out_lo)
    # A full-pool refit would actually differ here, or this test would pass
    # regardless of which pool the thresholds came from.
    assert row["threshold_hi"] != pytest.approx(full_hi)
    assert row["threshold_lo"] != pytest.approx(full_lo)


def test_thresholds_exclude_held_out_source_under_encoder_held_out(tmp_path):
    # Sanity check for the companion (non-fallback) path: when the test
    # source DOES carry both classes, thresholds still must not have used it.
    rng = np.random.default_rng(1)
    rows = []
    for src in ("s1", "s2", "s3"):
        for i in range(40):
            rows.append(_manifest_row(src, f"{src}_p{i}", float(rng.uniform(1, 200))))
    for i in range(10):
        rows.append(_manifest_row("s_test", f"s_test_lo{i}", float(rng.uniform(0, 2))))
    for i in range(10):
        rows.append(_manifest_row("s_test", f"s_test_hi{i}", float(rng.uniform(400, 500))))
    manifest = pd.DataFrame(rows)
    ssl_df, hand_df = _feature_frames(manifest, seed=1)

    project = _project(tmp_path, manifest, split_rows={
        "source_id": ["s1", "s2", "s3", "s_test"],
        "split": ["train", "train", "val", "test"],
    })

    probe = T.crystallinity_probe(project, "g", ssl_df, hand_df, METADATA_COLS, 0.85, 0.35)
    assert probe is not None
    row = probe.iloc[0]
    assert row["protocol"] == "encoder-held-out"

    fit_only_hi = manifest.loc[manifest.source_id != "s_test", "bragg_ratio"].quantile(0.85)
    fit_only_lo = manifest.loc[manifest.source_id != "s_test", "bragg_ratio"].quantile(0.35)
    assert row["threshold_hi"] == pytest.approx(fit_only_hi)
    assert row["threshold_lo"] == pytest.approx(fit_only_lo)


def test_held_out_source_never_enters_loso_fallback_evaluation(tmp_path):
    # A held-out source must never appear in the LOSO fallback's
    # eval_sources. This used to be reachable two ways: (a) a full-pool
    # threshold refit could incidentally give a single-class held-out source
    # both classes, sneaking it into "all-sources-encoder-exposed" (fixed by
    # reusing the non-held-out-only thresholds instead of refitting); (b) as
    # a defence regardless of where the thresholds came from, `df` itself
    # drops any checkpoint-designated held-out source before eval_sources is
    # computed. This test exercises (b) directly, independent of (a).
    rng = np.random.default_rng(3)
    rows = []
    for src in ("s1", "s2", "s3"):
        for i, v in enumerate(rng.uniform(90, 110, 15)):
            rows.append(_manifest_row(src, f"{src}_p{i}", float(v)))
    s_test_vals = np.concatenate([rng.uniform(0, 5, 10), rng.uniform(100, 106, 10)])
    for i, v in enumerate(s_test_vals):
        rows.append(_manifest_row("s_test", f"s_test_p{i}", float(v)))
    manifest = pd.DataFrame(rows)
    ssl_df, hand_df = _feature_frames(manifest, seed=3)

    # Confirm the fixture forces the fallback: under the non-held-out-only
    # thresholds that both the encoder-held-out attempt and (per the fix)
    # the fallback itself use, s_test must be single-class.
    fit_only = manifest.loc[manifest.source_id != "s_test", "bragg_ratio"]
    hi, lo = fit_only.quantile(0.85), fit_only.quantile(0.35)
    s_test_scores = manifest.loc[manifest.source_id == "s_test", "bragg_ratio"]
    labels = np.where(s_test_scores >= hi, 1, np.where(s_test_scores <= lo, 0, -1))
    assert set(labels[labels >= 0]) != {0, 1}, "fixture no longer forces the fallback"

    project = _project(tmp_path, manifest, split_rows={
        "source_id": ["s1", "s2", "s3", "s_test"],
        "split": ["train", "train", "val", "test"],
    })

    probe = T.crystallinity_probe(project, "g", ssl_df, hand_df, METADATA_COLS, 0.85, 0.35)
    assert probe is not None
    row = probe.iloc[0]
    assert row["protocol"] == "all-sources-encoder-exposed"
    eval_sources = row["eval_sources"].split(";")
    assert "s_test" not in eval_sources


def test_thresholds_use_full_pool_without_any_split(tmp_path):
    # No split_manifest.csv at all: this is the plain LOSO path, where using
    # the full pool was always correct and must remain so.
    rng = np.random.default_rng(2)
    rows = []
    for src in ("s1", "s2", "s3", "s4"):
        for i in range(30):
            rows.append(_manifest_row(src, f"{src}_p{i}", float(rng.uniform(1, 200))))
    manifest = pd.DataFrame(rows)
    ssl_df, hand_df = _feature_frames(manifest, seed=2)

    project = _project(tmp_path, manifest, split_rows=None)

    probe = T.crystallinity_probe(project, "g", ssl_df, hand_df, METADATA_COLS, 0.85, 0.35)
    assert probe is not None
    row = probe.iloc[0]
    assert row["protocol"] == "all-sources-encoder-exposed"

    full_hi = manifest["bragg_ratio"].quantile(0.85)
    full_lo = manifest["bragg_ratio"].quantile(0.35)
    assert row["threshold_hi"] == pytest.approx(full_hi)
    assert row["threshold_lo"] == pytest.approx(full_lo)
