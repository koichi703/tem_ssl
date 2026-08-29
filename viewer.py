#!/usr/bin/env python3
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from PIL import Image

import viewer_core as vc

st.set_page_config(
    page_title="TEM SSL Pilot Viewer",
    page_icon="🔬",
    layout="wide",
)
st.title("TEM Self-Supervised Learning Pilot Viewer")

default_project = (
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else Path("TEM_SSL_pilot")
)

project_text = st.sidebar.text_input(
    "Pilot project folder",
    str(default_project),
)
project = Path(project_text).expanduser()

manifest_path = project / "dataset" / "manifest.csv"
sources_path = project / "dataset" / "sources.csv"

if not manifest_path.exists():
    st.error(f"manifest.csv が見つかりません: {manifest_path}")
    st.stop()

manifest = pd.read_csv(manifest_path)
sources = pd.read_csv(sources_path) if sources_path.exists() else pd.DataFrame()

groups = sorted(manifest["scale_group"].dropna().astype(str).unique())
if not groups:
    st.error("scale_group が manifest.csv に見つかりません。")
    st.stop()

group = st.sidebar.selectbox(
    "Observation scale",
    groups,
    format_func=vc.scale_group_label,
)
st.sidebar.caption(vc.SCALE_GROUP_NOTE)

gdf = manifest[manifest["scale_group"].astype(str) == group].copy()

# --- Bragg score bands ------------------------------------------------
# Bands exist only when `analyze` has written usable thresholds. Without them
# the viewer still runs; it just shows the raw score, or nothing at all on an
# older dataset that predates the Bragg descriptor.
has_bragg = vc.has_bragg_scores(gdf)
thresholds = vc.read_bragg_thresholds(
    project / "analysis" / group / "crystallinity_probe.csv"
)
bands_on = bool(has_bragg and thresholds)

if bands_on:
    hi, lo = thresholds
    gdf[vc.BAND_COLUMN] = vc.classify_bragg_band(gdf["bragg_ratio"], hi, lo)
    selected_bands = st.sidebar.multiselect(
        "Bragg score band",
        list(vc.BAND_ORDER),
        default=list(vc.BAND_ORDER),
        help="Patches と PCA の両方に適用されます。",
    )
    st.sidebar.caption(vc.BRAGG_CAVEAT)
else:
    hi = lo = None
    selected_bands = []
    if not has_bragg:
        st.sidebar.info(
            "manifest.csv に bragg_ratio がありません。"
            " prepare を実行し直すと Bragg score が記録されます。"
        )
    else:
        st.sidebar.info(vc.probe_missing_note())


def apply_band_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a frame to the selected bands; a no-op when bands are off."""
    if not bands_on or vc.BAND_COLUMN not in df.columns:
        return df
    if not selected_bands:
        return df.iloc[0:0]
    return df[df[vc.BAND_COLUMN].isin(selected_bands)]


gdf_view = apply_band_filter(gdf)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Patches", "Bragg score", "Sources", "SSL PCA", "Handcrafted PCA"]
)


def series_to_arrow_safe_table(series: pd.Series) -> pd.DataFrame:
    """
    Convert a mixed-type pandas Series to a fully string-based two-column table.
    This avoids PyArrow errors caused by mixed str/numpy.int64/float values
    in one object-dtype column.
    """
    rows = []
    for key, value in series.items():
        if pd.isna(value):
            display_value = ""
        else:
            display_value = str(value)

        rows.append({
            "field": str(key),
            "value": display_value,
        })

    return pd.DataFrame(rows, dtype="string")


def dataframe_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve numeric columns where possible, but convert mixed object columns
    to strings for safe rendering in Streamlit/PyArrow.
    """
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else str(x)
            ).astype("string")

    return out


def metric_value(row: pd.Series, key: str, fmt: str) -> str:
    if key not in row.index or pd.isna(row[key]):
        return "—"
    try:
        return format(float(row[key]), fmt)
    except (TypeError, ValueError):
        return str(row[key])


with tab1:
    st.caption(f"Observation scale: **{vc.scale_group_label(group)}**")
    if bands_on and len(gdf_view) != len(gdf):
        st.caption(
            f"Bragg score band で絞り込み中: {len(gdf_view)} / {len(gdf)} パッチ"
        )

    source_ids = sorted(gdf_view["source_id"].dropna().astype(str).unique())

    if not source_ids:
        st.info(
            "選択中の条件に該当する source がありません。"
            if bands_on else "このscale groupにはsourceがありません。"
        )
    else:
        source_id = st.selectbox("Source", source_ids)

        sdf = gdf_view[gdf_view["source_id"].astype(str) == source_id].copy()
        patch_ids = sdf["patch_id"].dropna().astype(str).tolist()

        if not patch_ids:
            st.info("このsourceにはpatchがありません。")
        else:
            patch_id = st.selectbox("Patch", patch_ids)

            row = sdf[sdf["patch_id"].astype(str) == patch_id].iloc[0]

            c1, c2 = st.columns([1, 2])

            with c1:
                img_path = project / str(row["patch_path"])

                if img_path.exists():
                    st.image(
                        Image.open(img_path),
                        caption=patch_id,
                        width="stretch",
                    )
                else:
                    st.error(f"画像が見つかりません: {img_path}")

            with c2:
                if has_bragg:
                    st.subheader("Bragg score")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("bragg_ratio", metric_value(row, "bragg_ratio", ".2f"),
                              help="半径リング上の方位角 max/median を、ノイズ下の期待値で"
                                   " 割った値。等方ハローは 1 付近。")
                    m2.metric("bragg_d_nm", metric_value(row, "bragg_d_nm", ".4f"),
                              help="スコアが立ったリングの格子縞間隔 (nm)。")
                    m3.metric("bragg_pixels", metric_value(row, "bragg_pixels", ".0f"),
                              help="リング中央値の 5 倍を超えた画素数。")
                    if bands_on:
                        band = row.get(vc.BAND_COLUMN)
                        st.write(f"Band: **{band if pd.notna(band) else vc.BAND_UNSCORED}**")
                    st.caption(vc.BRAGG_CAVEAT)

                st.subheader("Patch metadata")

                metadata_table = series_to_arrow_safe_table(row)
                st.dataframe(
                    metadata_table,
                    width="stretch",
                    height=520,
                    hide_index=True,
                )

            nn_path = (
                project
                / "analysis"
                / group
                / "cross_source_nearest_neighbors.csv"
            )

            if nn_path.exists():
                nn = pd.read_csv(nn_path)
                nn = nn[
                    nn["query_patch_id"].astype(str) == patch_id
                ].copy()

                if not nn.empty:
                    st.subheader("Cross-source nearest neighbours")

                    nn = nn.head(5)
                    cols = st.columns(len(nn))

                    for col, (_, nrow) in zip(cols, nn.iterrows()):
                        match = manifest[
                            manifest["patch_id"].astype(str)
                            == str(nrow["neighbor_patch_id"])
                        ]

                        with col:
                            if not match.empty:
                                npath = project / str(
                                    match.iloc[0]["patch_path"]
                                )

                                if npath.exists():
                                    st.image(
                                        Image.open(npath),
                                        width="stretch",
                                    )

                                try:
                                    cos = float(nrow["cosine_similarity"])
                                    cos_text = f"{cos:.3f}"
                                except Exception:
                                    cos_text = str(
                                        nrow["cosine_similarity"]
                                    )

                                st.caption(
                                    f"{nrow['neighbor_patch_id']}\n"
                                    f"cos={cos_text}"
                                )


with tab2:
    if not has_bragg:
        st.info(
            "manifest.csv に bragg_ratio がありません。"
            " prepare を実行し直すと Bragg score が記録されます。"
        )
    else:
        st.caption(vc.BRAGG_CAVEAT)

        if bands_on:
            st.write(
                f"閾値: High ≥ **{hi:.4g}** / Low ≤ **{lo:.4g}** "
                f"(analysis/{group}/crystallinity_probe.csv)"
            )
        else:
            st.info(vc.probe_missing_note())

        hist = vc.bragg_histogram(gdf["bragg_ratio"])
        if hist.empty:
            st.info("ヒストグラムを描ける有限値がありません。")
        else:
            st.subheader("Bragg score の分布")
            st.caption("横軸は対数スケール（スコアは大きく右に裾を引きます）。")
            st.bar_chart(hist, x="bragg_ratio", y="patches", width="stretch")

        if bands_on:
            st.subheader("Band ごとの patch 数 / source 数")
            st.dataframe(
                dataframe_arrow_safe(vc.band_summary(gdf)),
                width="stretch", hide_index=True,
            )

            by_source = vc.band_by_source(gdf)
            if not by_source.empty:
                st.subheader("Source ごとの band 構成")
                st.caption(
                    "学習サンプラーを変更するかどうかは、まずこの偏りを見てから判断してください。"
                )
                st.dataframe(
                    dataframe_arrow_safe(by_source),
                    width="stretch", hide_index=True,
                )


with tab3:
    if sources.empty:
        st.info("sources.csv がありません。")
    else:
        filtered_sources = sources[
            sources["scale_group"].astype(str) == group
        ].copy()

        st.dataframe(
            dataframe_arrow_safe(filtered_sources),
            width="stretch",
            height=600,
        )


def pca_tab(path: Path):
    if not path.exists():
        st.info(
            "まだ解析結果がありません。"
            " extract → analyze を実行してください。"
        )
        return

    df = pd.read_csv(path)
    df = vc.attach_bragg_columns(df, manifest)

    color_col = None
    if bands_on and "bragg_ratio" in df.columns:
        df[vc.BAND_COLUMN] = vc.classify_bragg_band(df["bragg_ratio"], hi, lo)
        df = apply_band_filter(df)
        color_col = vc.BAND_COLUMN
        st.caption(vc.BRAGG_CAVEAT)

    st.dataframe(
        dataframe_arrow_safe(df),
        width="stretch",
        height=500,
    )

    if "PC1" in df.columns and "PC2" in df.columns:
        keep = ["PC1", "PC2"] + ([color_col] if color_col else [])
        plot_df = df[keep].copy()

        plot_df["PC1"] = pd.to_numeric(
            plot_df["PC1"],
            errors="coerce",
        )
        plot_df["PC2"] = pd.to_numeric(
            plot_df["PC2"],
            errors="coerce",
        )
        plot_df = plot_df.dropna(subset=["PC1", "PC2"])

        if color_col:
            plot_df[color_col] = plot_df[color_col].fillna(vc.BAND_UNSCORED).astype(str)

        if not plot_df.empty:
            st.scatter_chart(
                plot_df,
                x="PC1",
                y="PC2",
                color=color_col,
                width="stretch",
            )


with tab4:
    pca_tab(
        project
        / "analysis"
        / group
        / "pca_ssl.csv"
    )


with tab5:
    pca_tab(
        project
        / "analysis"
        / group
        / "pca_handcrafted.csv"
    )
