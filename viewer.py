#!/usr/bin/env python3
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from PIL import Image

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

group = st.sidebar.selectbox("Scale group", groups)
gdf = manifest[manifest["scale_group"].astype(str) == group].copy()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Patches", "Sources", "SSL PCA", "Handcrafted PCA"]
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


with tab1:
    source_ids = sorted(gdf["source_id"].dropna().astype(str).unique())

    if not source_ids:
        st.info("このscale groupにはsourceがありません。")
    else:
        source_id = st.selectbox("Source", source_ids)

        sdf = gdf[gdf["source_id"].astype(str) == source_id].copy()
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
                st.subheader("Patch metadata")

                metadata_table = series_to_arrow_safe_table(row)
                st.dataframe(
                    metadata_table,
                    width="stretch",
                    height=650,
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

    st.dataframe(
        dataframe_arrow_safe(df),
        width="stretch",
        height=500,
    )

    if "PC1" in df.columns and "PC2" in df.columns:
        plot_df = df[["PC1", "PC2"]].copy()

        plot_df["PC1"] = pd.to_numeric(
            plot_df["PC1"],
            errors="coerce",
        )
        plot_df["PC2"] = pd.to_numeric(
            plot_df["PC2"],
            errors="coerce",
        )
        plot_df = plot_df.dropna()

        if not plot_df.empty:
            st.scatter_chart(
                plot_df,
                x="PC1",
                y="PC2",
                width="stretch",
            )


with tab3:
    pca_tab(
        project
        / "analysis"
        / group
        / "pca_ssl.csv"
    )


with tab4:
    pca_tab(
        project
        / "analysis"
        / group
        / "pca_handcrafted.csv"
    )
