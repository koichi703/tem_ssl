"""Tests for the viewer's pure helpers (no Streamlit involved)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import viewer_core as vc


# ---------------------------------------------------------------- labels
def test_known_scale_groups_get_observation_scale_labels():
    assert vc.scale_group_label("lattice") == "High-resolution / atomic scale (lattice)"
    for key in ("nano", "meso", "micro"):
        label = vc.scale_group_label(key)
        assert key in label and label != key


def test_unknown_and_empty_scale_groups_do_not_raise():
    assert vc.scale_group_label("custom") == "Observation scale (custom)"
    assert vc.scale_group_label(None) == "Observation scale"


def test_labels_never_claim_a_phase():
    joined = " ".join(vc.SCALE_GROUP_LABELS.values()).lower()
    for word in ("crystal", "amorphous", "結晶", "非晶"):
        assert word not in joined


def test_band_names_are_score_bands_not_phase_labels():
    joined = " ".join(vc.BAND_ORDER).lower()
    assert "crystalline" not in joined and "amorphous" not in joined


# ------------------------------------------------------------ thresholds
def _probe(tmp_path, **cols):
    p = tmp_path / "crystallinity_probe.csv"
    pd.DataFrame(cols).to_csv(p, index=False)
    return p


def test_reads_valid_thresholds(tmp_path):
    p = _probe(tmp_path, representation=["ssl"], threshold_hi=[29.9], threshold_lo=[2.3])
    assert vc.read_bragg_thresholds(p) == (29.9, 2.3)


def test_missing_file_returns_none(tmp_path):
    assert vc.read_bragg_thresholds(tmp_path / "nope.csv") is None


def test_empty_file_returns_none(tmp_path):
    p = tmp_path / "crystallinity_probe.csv"
    p.write_text("")
    assert vc.read_bragg_thresholds(p) is None


def test_corrupt_file_returns_none(tmp_path):
    p = tmp_path / "crystallinity_probe.csv"
    p.write_text("not,a\nvalid\x00csv,,,,\n\"unterminated")
    assert vc.read_bragg_thresholds(p) is None


def test_missing_columns_returns_none(tmp_path):
    assert vc.read_bragg_thresholds(_probe(tmp_path, auc_mean=[0.9])) is None


@pytest.mark.parametrize("hi,lo", [
    (np.nan, 2.0), (2.0, np.nan), (np.inf, 2.0), (-np.inf, np.inf),
    (2.0, 2.0),          # not separable
    (1.0, 9.0),          # inverted
])
def test_non_finite_or_inverted_thresholds_rejected(tmp_path, hi, lo):
    assert vc.read_bragg_thresholds(
        _probe(tmp_path, threshold_hi=[hi], threshold_lo=[lo])) is None


def test_skips_leading_nan_rows(tmp_path):
    p = _probe(tmp_path, threshold_hi=[np.nan, 30.0], threshold_lo=[np.nan, 2.0])
    assert vc.read_bragg_thresholds(p) == (30.0, 2.0)


def test_directory_path_returns_none(tmp_path):
    assert vc.read_bragg_thresholds(tmp_path) is None


# ---------------------------------------------------------- band mapping
def test_classify_splits_at_the_thresholds():
    band = vc.classify_bragg_band([100.0, 10.0, 0.5, 30.0, 2.0], hi=30.0, lo=2.0)
    assert list(band) == [vc.BAND_HIGH, vc.BAND_AMBIGUOUS, vc.BAND_LOW,
                          vc.BAND_HIGH, vc.BAND_LOW]


def test_classify_leaves_nan_and_text_unbanded():
    band = vc.classify_bragg_band([np.nan, "x", None, np.inf, 50.0], hi=30.0, lo=2.0)
    assert list(band.isna()) == [True, True, True, True, False]
    assert band.iloc[4] == vc.BAND_HIGH


def test_classify_preserves_a_non_default_index():
    s = pd.Series([100.0, 0.1], index=["a", "b"])
    assert list(vc.classify_bragg_band(s, 30.0, 2.0).index) == ["a", "b"]


@pytest.mark.parametrize("hi,lo", [(np.nan, 2.0), (2.0, 5.0), ("x", 2.0), (None, None)])
def test_classify_with_bad_thresholds_yields_all_na(hi, lo):
    band = vc.classify_bragg_band([1.0, 2.0, 3.0], hi, lo)
    assert band.isna().all()


def test_classify_on_empty_and_constant_input():
    assert vc.classify_bragg_band([], 30.0, 2.0).empty
    band = vc.classify_bragg_band([5.0] * 4, 30.0, 2.0)
    assert set(band) == {vc.BAND_AMBIGUOUS}


# --------------------------------------------------------- score presence
def test_has_bragg_scores():
    assert vc.has_bragg_scores(pd.DataFrame({"bragg_ratio": [1.0, np.nan]}))
    assert not vc.has_bragg_scores(pd.DataFrame({"bragg_ratio": [np.nan, np.nan]}))
    assert not vc.has_bragg_scores(pd.DataFrame({"other": [1.0]}))
    assert not vc.has_bragg_scores(None)


# ------------------------------------------------------------------ join
def _manifest():
    return pd.DataFrame({
        "patch_id": ["a", "b", "c"],
        "source_id": ["s1", "s1", "s2"],
        "bragg_ratio": [100.0, 1.0, np.nan],
        "bragg_d_nm": [0.19, np.nan, np.nan],
        "bragg_pixels": [12.0, 0.0, 0.0],
    })


def test_attach_joins_by_patch_id_without_duplicating_rows():
    pca = pd.DataFrame({"patch_id": ["c", "a"], "PC1": [1.0, 2.0]})
    out = vc.attach_bragg_columns(pca, _manifest())
    assert len(out) == 2
    assert list(out["patch_id"]) == ["c", "a"]
    assert out.loc[out.patch_id == "a", "bragg_ratio"].iloc[0] == 100.0
    assert pd.isna(out.loc[out.patch_id == "c", "bragg_ratio"].iloc[0])


def test_attach_is_a_noop_without_the_columns_or_key():
    pca = pd.DataFrame({"patch_id": ["a"], "PC1": [1.0]})
    old = pd.DataFrame({"patch_id": ["a"], "scale_group": ["lattice"]})
    assert list(vc.attach_bragg_columns(pca, old).columns) == ["patch_id", "PC1"]
    nokey = pd.DataFrame({"PC1": [1.0]})
    assert list(vc.attach_bragg_columns(nokey, _manifest()).columns) == ["PC1"]
    assert vc.attach_bragg_columns(None, _manifest()) is None
    assert vc.attach_bragg_columns(pca, None) is pca


def test_attach_replaces_stale_bragg_columns_rather_than_suffixing():
    pca = pd.DataFrame({"patch_id": ["a"], "bragg_ratio": [-1.0]})
    out = vc.attach_bragg_columns(pca, _manifest())
    assert "bragg_ratio_x" not in out.columns
    assert out["bragg_ratio"].iloc[0] == 100.0


def test_attach_survives_duplicate_manifest_rows():
    dup = pd.concat([_manifest(), _manifest()], ignore_index=True)
    pca = pd.DataFrame({"patch_id": ["a", "b"], "PC1": [1.0, 2.0]})
    assert len(vc.attach_bragg_columns(pca, dup)) == 2


# -------------------------------------------------------------- summaries
def _banded():
    m = _manifest()
    m[vc.BAND_COLUMN] = vc.classify_bragg_band(m["bragg_ratio"], 30.0, 2.0)
    return m


def test_band_summary_counts_patches_and_sources_in_fixed_order():
    out = vc.band_summary(_banded())
    assert list(out["band"])[:3] == list(vc.BAND_ORDER)
    high = out[out.band == vc.BAND_HIGH].iloc[0]
    assert (high.patches, high.sources) == (1, 1)
    assert out[out.band == vc.BAND_UNSCORED].iloc[0].patches == 1


def test_band_summary_without_a_band_column_is_empty_not_an_error():
    assert vc.band_summary(pd.DataFrame({"patch_id": ["a"]})).empty
    assert vc.band_summary(None).empty


def test_band_by_source_reports_every_band_column():
    out = vc.band_by_source(_banded())
    assert "source_id" in out.columns and len(out) == 2
    assert vc.BAND_UNSCORED in out.columns


def test_band_by_source_degrades_quietly():
    assert vc.band_by_source(pd.DataFrame({"patch_id": ["a"]})).empty
    assert vc.band_by_source(None).empty


# -------------------------------------------------------------- histogram
def test_histogram_bins_finite_positive_values():
    h = vc.bragg_histogram([1.0, 10.0, 100.0], bins=8)
    assert list(h.columns) == ["bragg_ratio", "patches"]
    assert h["patches"].sum() == 3


def test_histogram_drops_nan_inf_and_non_positive_on_log():
    h = vc.bragg_histogram([np.nan, np.inf, -np.inf, 0.0, -5.0, 4.0], bins=4)
    assert h["patches"].sum() == 1


def test_histogram_on_empty_or_all_invalid_returns_empty_frame():
    for bad in ([], [np.nan, np.nan], [0.0, -1.0], ["x", None]):
        h = vc.bragg_histogram(bad)
        assert h.empty and list(h.columns) == ["bragg_ratio", "patches"]


def test_histogram_handles_constant_values():
    h = vc.bragg_histogram([7.0] * 5, bins=6)
    assert h["patches"].sum() == 5


def test_histogram_linear_mode_keeps_non_positive_values():
    h = vc.bragg_histogram([-1.0, 0.0, 1.0], bins=4, log=False)
    assert h["patches"].sum() == 3
