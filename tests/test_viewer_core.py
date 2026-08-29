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
    assert list(h.columns) == ["bragg_ratio", "patches", "bin_lo", "bin_hi", "bin_id"]
    assert h["patches"].sum() == 3


def test_histogram_drops_nan_inf_and_non_positive_on_log():
    h = vc.bragg_histogram([np.nan, np.inf, -np.inf, 0.0, -5.0, 4.0], bins=4)
    assert h["patches"].sum() == 1


def test_histogram_on_empty_or_all_invalid_returns_empty_frame():
    for bad in ([], [np.nan, np.nan], [0.0, -1.0], ["x", None]):
        h = vc.bragg_histogram(bad)
        assert h.empty and list(h.columns) == ["bragg_ratio", "patches", "bin_lo", "bin_hi", "bin_id"]


def test_histogram_handles_constant_values():
    h = vc.bragg_histogram([7.0] * 5, bins=6)
    assert h["patches"].sum() == 5


def test_histogram_linear_mode_keeps_non_positive_values():
    h = vc.bragg_histogram([-1.0, 0.0, 1.0], bins=4, log=False)
    assert h["patches"].sum() == 3


def test_histogram_preserves_distinct_centers_on_a_narrow_range():
    # Codex-reported case: a range under ~1e-4 collapsed every bin center to
    # the same value (or to 0) under a fixed 4-decimal rounding.
    h = vc.bragg_histogram([1.0, 1.00001], bins=8, log=False)
    assert h["bragg_ratio"].nunique() == len(h)
    assert (h["bragg_ratio"] != 0).all() or h["patches"].sum() == 2


def test_histogram_still_rounds_to_four_decimals_on_an_ordinary_range():
    h = vc.bragg_histogram([1.0, 500.0], bins=10)
    # readability is unaffected when nothing needs extra precision
    assert all(round(v, 4) == v for v in h["bragg_ratio"])


def test_histogram_widens_precision_only_as_far_as_needed():
    h = vc.bragg_histogram([100.0, 100.0 + 1e-8], bins=4, log=False)
    assert h["bragg_ratio"].nunique() == len(h)


def test_histogram_falls_back_to_unrounded_labels_past_the_precision_cap():
    # Codex-reported case: spacing near float64 epsilon meant no decimal
    # count up to the cap kept all centers distinct, so the old code rounded
    # to `cap` anyway and re-collapsed bins that were distinct unrounded.
    h = vc.bragg_histogram([1.0, 1.0 + 2e-14], bins=40)
    assert h["bragg_ratio"].nunique() == len(h) == 40


def test_distinguishing_decimals_returns_none_past_the_cap():
    import numpy as np
    values = np.array([1.0, 1.0 + 1e-16])
    assert vc._distinguishing_decimals(values, start=4, cap=15) is None


def test_distinguishing_decimals_single_value():
    import numpy as np
    assert vc._distinguishing_decimals(np.array([1.0]), start=4) == 4


def test_histogram_never_rounds_a_positive_center_to_zero():
    # Codex-reported case: 4-decimal rounding turned a finite positive center
    # (~5.6e-9) into 0.0, which was merely "unique" from its neighbour, not
    # a value that ever belonged to the histogram.
    h = vc.bragg_histogram([1e-10, 1e-3], bins=2)
    assert (h["bragg_ratio"] > 0).all()
    assert h["bragg_ratio"].nunique() == len(h) == 2


def test_histogram_keeps_extreme_finite_centers_finite():
    h = vc.bragg_histogram([1e300, 1e301], bins=3)
    assert np.isfinite(h["bragg_ratio"]).all()
    assert h["bragg_ratio"].nunique() == len(h) == 3


def test_distinguishing_decimals_rejects_a_round_to_zero():
    values = np.array([5.6e-9, 1.78e-5])
    d = vc._distinguishing_decimals(values, start=4)
    assert d is not None
    assert not np.any(np.round(values, d) == 0)


def test_histogram_singleton_does_not_round_a_positive_center_to_zero():
    # Codex-reported case: the n <= 1 early return skipped the finite/nonzero
    # checks entirely, so a one-bin histogram of a tiny positive score
    # rounded straight to 0.0 with nothing to catch it.
    h = vc.bragg_histogram([1e-10], bins=1)
    assert len(h) == 1
    assert (h["bragg_ratio"] > 0).all()


def test_histogram_singleton_keeps_an_extreme_center_finite():
    h = vc.bragg_histogram([1e300], bins=1)
    assert len(h) == 1
    assert np.isfinite(h["bragg_ratio"]).all()


def test_distinguishing_decimals_singleton_widens_precision_when_needed():
    assert vc._distinguishing_decimals(np.array([1e-10]), start=4) == 10
    assert vc._distinguishing_decimals(np.array([5.0]), start=4) == 4


def test_distinguishing_decimals_empty_array_is_harmless():
    assert vc._distinguishing_decimals(np.array([]), start=4) == 4


# ------------------------------------------------------ selection -> patches
def _scatter_df():
    return pd.DataFrame({
        "patch_id": ["a", "b", "c", "d"],
        "PC1": [0.0, 1.0, 2.0, np.nan],
        "PC2": [0.0, 1.0, 5.0, 1.0],
    })


def test_patches_in_rect_selects_the_covered_points():
    ids = vc.patches_in_rect(_scatter_df(), "PC1", "PC2", (-0.5, 1.5), (-0.5, 1.5))
    assert ids == ["a", "b"]


def test_patches_in_rect_accepts_reversed_bounds():
    ids = vc.patches_in_rect(_scatter_df(), "PC1", "PC2", (1.5, -0.5), (1.5, -0.5))
    assert ids == ["a", "b"]


def test_patches_in_rect_excludes_nan_coordinates():
    ids = vc.patches_in_rect(_scatter_df(), "PC1", "PC2", (-10, 10), (-10, 10))
    assert "d" not in ids and set(ids) == {"a", "b", "c"}


@pytest.mark.parametrize("x_range,y_range", [
    ((np.nan, 1.0), (0.0, 1.0)),
    ((0.0, np.inf), (0.0, 1.0)),
    (("x", "y"), (0.0, 1.0)),
])
def test_patches_in_rect_bad_range_returns_empty_not_raise(x_range, y_range):
    assert vc.patches_in_rect(_scatter_df(), "PC1", "PC2", x_range, y_range) == []


def test_patches_in_rect_missing_columns_or_frame():
    assert vc.patches_in_rect(pd.DataFrame({"patch_id": ["a"]}), "PC1", "PC2", (0, 1), (0, 1)) == []
    assert vc.patches_in_rect(None, "PC1", "PC2", (0, 1), (0, 1)) == []


def _score_df():
    return pd.DataFrame({
        "patch_id": ["a", "b", "c", "d"],
        "bragg_ratio": [1.0, 5.0, np.nan, 10.0],
    })


def test_patches_in_score_range_is_inclusive_on_both_ends():
    ids = vc.patches_in_score_range(_score_df(), "bragg_ratio", 1.0, 5.0)
    assert ids == ["a", "b"]


def test_patches_in_score_range_excludes_nan():
    assert "c" not in vc.patches_in_score_range(_score_df(), "bragg_ratio", 0.0, 100.0)


@pytest.mark.parametrize("lo,hi", [(np.nan, 5.0), (1.0, np.nan), (5.0, 1.0)])
def test_patches_in_score_range_bad_bounds_returns_empty(lo, hi):
    assert vc.patches_in_score_range(_score_df(), "bragg_ratio", lo, hi) == []


def test_patches_in_score_range_missing_columns_or_frame():
    assert vc.patches_in_score_range(pd.DataFrame({"patch_id": ["a"]}), "bragg_ratio", 0, 1) == []
    assert vc.patches_in_score_range(None, "bragg_ratio", 0, 1) == []


# -------------------------------------------------------------- path lookup
def _manifest_paths():
    return pd.DataFrame({
        "patch_id": ["a", "b", "b"],  # duplicate id, first wins
        "patch_path": ["x/a.png", "x/b.png", "x/b2.png"],
    })


def test_resolve_patch_paths_joins_in_requested_order():
    out = vc.resolve_patch_paths(["b", "a"], _manifest_paths(), "/proj")
    assert [pid for pid, _ in out] == ["b", "a"]
    assert dict(out)["a"] == Path("/proj/x/a.png")


def test_resolve_patch_paths_drops_unknown_ids_silently():
    out = vc.resolve_patch_paths(["a", "nope"], _manifest_paths(), "/proj")
    assert [pid for pid, _ in out] == ["a"]


def test_resolve_patch_paths_empty_inputs():
    assert vc.resolve_patch_paths([], _manifest_paths(), "/proj") == []
    assert vc.resolve_patch_paths(["a"], None, "/proj") == []
    assert vc.resolve_patch_paths(["a"], pd.DataFrame({"patch_id": ["a"]}), "/proj") == []


# ------------------------------------------------------- histogram bin edges
def test_histogram_bin_bounds_cover_the_reported_center():
    h = vc.bragg_histogram([1.0, 10.0, 100.0], bins=8)
    assert (h["bin_lo"] <= h["bin_hi"]).all()
    assert list(h["bin_id"]) == list(range(len(h)))


def test_histogram_bin_bounds_are_not_truncated_by_display_rounding():
    # bin_lo/bin_hi drive the click-filter; they must stay at full precision
    # even when the displayed center needed the > 4 decimal fallback.
    h = vc.bragg_histogram([1.0, 1.00001], bins=8, log=False)
    assert not (h["bin_lo"] == h["bin_hi"]).any()


# ------------------------------------------------------------ regression
def test_bragg_histogram_bin_id_matches_patches_in_score_range():
    """End-to-end shape check: a clicked bin's bounds actually recover its count."""
    values = pd.Series([1.0, 2.0, 2.0, 50.0, 200.0])
    ids = pd.Series(["a", "b", "c", "d", "e"])
    df = pd.DataFrame({"patch_id": ids, "bragg_ratio": values})
    hist = vc.bragg_histogram(values, bins=5)
    for _, row in hist.iterrows():
        selected = vc.patches_in_score_range(df, "bragg_ratio", row["bin_lo"], row["bin_hi"])
        in_range = df[(df.bragg_ratio >= row["bin_lo"]) & (df.bragg_ratio <= row["bin_hi"])]
        assert set(selected) == set(in_range.patch_id)
