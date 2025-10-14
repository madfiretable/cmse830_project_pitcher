# app.py — Streamlit: IDA/EDA, pre/post-imputation heatmap with tuning,
# KDE/Violin/Scatter, download. (No sticker/overlay code)
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# --------------------------- Page/UI config ---------------------------
st.set_page_config(page_title="Pitching Analytics (MLB + CPBL)", layout="wide")
st.title("Pitching Analytics (MLB + CPBL)")

# --------------------------- Sidebar controls ------------------------
st.sidebar.header("Settings")

# Data filtering
BF_THRESHOLD = st.sidebar.number_input(
    "BF threshold (drop if BF < threshold)", min_value=0, value=70, step=5
)

# EDA toggles
st.sidebar.subheader("EDA Controls")
show_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
exclude_bf_in_heatmap = st.sidebar.checkbox("Exclude BF from Heatmap", value=True)
show_used_cols = st.sidebar.checkbox("Show heatmap variable list", value=False)

# Heatmap tuning (NEW)
st.sidebar.subheader("Heatmap Tuning")
heatmap_scale = st.sidebar.slider("Cell size scale", 0.6, 1.8, 1.2, 0.1)
heatmap_text = st.sidebar.slider("Text size", 6, 18, 11, 1)

# Visual toggles
st.sidebar.subheader("Visualizations")
show_kde = st.sidebar.checkbox("Show KDE by League", value=True)
show_violin = st.sidebar.checkbox("Show Team Violin Plot", value=True)
violin_metric = st.sidebar.selectbox(
    "Violin metric (by Team)",
    ["ERA_plus", "ERA", "FIP", "WHIP", "K%", "BB%", "WAR", "HR9", "BAbip"],
    index=0,
)
facet_violin_by_league = st.sidebar.checkbox("Facet Violin by League", value=True)
show_scatter = st.sidebar.checkbox("Show ERA+ vs Others (Scatter Grid)", value=True)
max_scatter = st.sidebar.slider("Max scatter panels", 1, 20, 12)

# Styling
st.sidebar.subheader("Style")
sns.set_theme(style="whitegrid")

# --------------------------- Data paths -------------------------------
MLB_PATH = "MLB_Pitch.csv"
CPBL_PATH = "投手.xlsx"

def assert_file(path: str):
    if not os.path.exists(path):
        st.error(f"File not found: `{path}`. Put it in the same folder as this app.")
        st.stop()

assert_file(MLB_PATH)
assert_file(CPBL_PATH)

# --------------------------- Load raw data ---------------------------
mlb_raw = pd.read_csv(MLB_PATH)
cpbl_raw = pd.read_excel(CPBL_PATH)

# --------------------------- Helpers ---------------------------------
def prepare_and_merge(mlb_df: pd.DataFrame, cpbl_df: pd.DataFrame, bf_threshold: int):
    """
    Align schema; compute CPBL SO/BB; drop by BF; keep final columns (including BAbip);
    return combined (clean, before imputation) and artifacts.
    """
    cpbl = cpbl_df.copy()
    mlb = mlb_df.copy()

    # 1) SO/BB in CPBL
    if ("K%" in cpbl.columns) and ("BB%" in cpbl.columns):
        cpbl["SO/BB"] = (cpbl["K%"] / cpbl["BB%"]).replace([np.inf, -np.inf], np.nan)

    # 2) CPBL rename to align
    rename_map_cpbl = {
        "球員": "Player",
        "背號": "No",
        "球隊": "Team",
        "AVG": "BA",
        "BABIP": "BAbip",
        "ERA+": "ERA_plus",
        "FIP": "FIP",
        "WHIP": "WHIP",
        "OPS": "OPS",
        "K%": "K%",
        "BB%": "BB%",
        "Whiff%": "Whiff%",
        "Swing%": "Swing%",
        "PutAway%": "PutAway%",
        "GB%": "GB%",
        "FB%": "FB%",
    }
    cpbl = cpbl.rename(columns=rename_map_cpbl)

    # 3) MLB rename to align
    mlb = mlb.rename(columns={"ERA+": "ERA_plus"})
    if "Team" not in mlb.columns:
        if "Tm" in mlb.columns:
            mlb = mlb.rename(columns={"Tm": "Team"})
        else:
            mlb["Team"] = "Unknown"

    # 4) Drop by BF
    for df in (mlb, cpbl):
        df["BF"] = pd.to_numeric(df.get("BF", np.nan), errors="coerce")

    mlb_before, cpbl_before = len(mlb), len(cpbl)
    mlb_clean = mlb.loc[mlb["BF"].replace([np.inf, -np.inf], np.nan) >= bf_threshold].copy()
    cpbl_clean = cpbl.loc[cpbl["BF"].replace([np.inf, -np.inf], np.nan) >= bf_threshold].copy()
    mlb_after, cpbl_after = len(mlb_clean), len(cpbl_clean)

    # 5) Final columns (include BAbip)
    common_cols = [
        "Player", "Team", "IP", "BF", "ERA", "ERA_plus", "FIP", "WHIP",
        "K%", "BB%", "GB%", "FB%", "SO/BB", "BAbip"
    ]
    mlb_unique  = ["WAR", "HR9"]
    cpbl_unique = ["Whiff%", "Swing%", "PutAway%"]
    final_cols  = common_cols + mlb_unique + cpbl_unique

    # Ensure structure
    for col in final_cols:
        if col not in mlb_clean.columns:  mlb_clean[col]  = np.nan
        if col not in cpbl_clean.columns: cpbl_clean[col] = np.nan

    mlb_clean  = mlb_clean[final_cols].copy();  mlb_clean["League"]  = "MLB"
    cpbl_clean = cpbl_clean[final_cols].copy(); cpbl_clean["League"] = "CPBL"

    combined_clean_no_impute = pd.concat([mlb_clean, cpbl_clean], ignore_index=True)

    artifacts = {
        "counts": {
            "mlb_before": mlb_before, "mlb_after": mlb_after,
            "cpbl_before": cpbl_before, "cpbl_after": cpbl_after
        },
        "mlb_clean": mlb_clean, "cpbl_clean": cpbl_clean
    }
    return combined_clean_no_impute, artifacts

def impute_all(combined_clean_no_impute: pd.DataFrame) -> pd.DataFrame:
    """
    1) KNNImputer to impute WAR (features include BAbip)
    2) IterativeImputer across all numeric columns with Team/League encodings
    """
    combined = combined_clean_no_impute.copy()

    # KNN for WAR
    ref_cols = [c for c in ["ERA","FIP","WHIP","K%","BB%","SO/BB","HR9","ERA_plus","BAbip"]
                if c in combined.columns]
    if "WAR" in combined.columns and len(ref_cols) > 0:
        knn_data = combined[ref_cols + ["WAR"]].copy()
        knn_imp  = KNNImputer(n_neighbors=8, weights="distance")
        imputed  = knn_imp.fit_transform(knn_data)
        combined["WAR"] = imputed[:, -1]

    # IterativeImputer (team/league aware)
    combined["Team_code"]   = pd.factorize(combined["Team"])[0].astype(float)
    combined["League_code"] = (combined["League"] == "MLB").astype(float)

    numeric_cols = combined.select_dtypes(include=["float64", "int64"]).columns.tolist()
    X = combined[numeric_cols].replace([np.inf, -np.inf], np.nan)

    valid_cols = [c for c in X.columns if X[c].notna().sum() > 0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[valid_cols])

    imp = IterativeImputer(max_iter=30, random_state=42, tol=1e-3)
    X_imp_scaled = imp.fit_transform(X_scaled)
    X_imp = scaler.inverse_transform(X_imp_scaled)

    X_filled = X.copy()
    X_filled[valid_cols] = X_imp
    combined[numeric_cols] = X_filled[numeric_cols]

    # Clip logic
    for c in [c for c in combined.columns if c.endswith("%")]:
        combined[c] = combined[c].clip(0, 100)
    if "HR9" in combined.columns:
        combined["HR9"] = combined["HR9"].clip(lower=0)

    return combined

def describe_by_dtype(df: pd.DataFrame):
    """Return numeric and categorical summaries."""
    num_cols = df.select_dtypes(include=["float64","int64"]).columns
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns

    desc_num = df[num_cols].describe().T if len(num_cols) else pd.DataFrame()
    if len(cat_cols):
        nunique = df[cat_cols].nunique().rename("nunique")
        top = df[cat_cols].agg(lambda s: s.value_counts(dropna=False).index[0] if s.count() else np.nan).rename("top")
        freq = df[cat_cols].agg(lambda s: s.value_counts(dropna=False).iloc[0] if s.count() else np.nan).rename("freq")
        desc_cat = pd.concat([nunique, top, freq], axis=1)
    else:
        desc_cat = pd.DataFrame()

    return desc_num, desc_cat

def missing_table(df: pd.DataFrame):
    nulls = df.isna().sum()
    pct = (nulls / len(df) * 100).round(2)
    out = pd.DataFrame({"missing": nulls, "missing_%": pct})
    return out[out["missing"] > 0].sort_values("missing_%", ascending=False)

# --------------------------- Pipeline run ----------------------------
combined_clean_no_impute, art = prepare_and_merge(mlb_raw, cpbl_raw, BF_THRESHOLD)
combined_imputed = impute_all(combined_clean_no_impute)

# Sidebar kept counts
st.sidebar.markdown("---")
st.sidebar.write(
    f"MLB kept: {art['counts']['mlb_after']}/{art['counts']['mlb_before']} | "
    f"CPBL kept: {art['counts']['cpbl_after']}/{art['counts']['cpbl_before']}"
)

# --------------------------- Tabs -----------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "IDA (Raw)", "EDA (Clean/Imputed)", "Visualizations", "Download"
])

# --------------------------- Overview -------------------------------
with tab1:
    st.subheader("Data sources and pipeline")
    st.markdown("""
- **Sources:**  
  - **MLB:**  
    [Standard Pitching (MLB)](https://www.baseball-reference.com/leagues/majors/2025-standard-pitching.shtml),  
    [Advanced Pitching (MLB)](https://www.baseball-reference.com/leagues/majors/2025-advanced-pitching.shtml)  
  - **CPBL (中華職棒):**  
    [Brothers](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/Kae1X-%E4%B8%AD%E4%BF%A1%E5%85%84%E5%BC%9F?tab=pitching) |  
    [Hawks](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/t6zJf-%E5%8F%B0%E9%8B%BC%E9%9B%84%E9%B7%B9?tab=pitching) |  
    [Dragons](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/R2VRh-%E5%91%B3%E5%85%A8%E9%BE%8D?tab=pitching) |  
    [Guardians](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/wi4T3-%E5%AF%8C%E9%82%A6%E6%82%8D%E5%B0%87?tab=pitching) |  
    [Monkeys](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/WyADE-%E6%A8%82%E5%A4%A9%E6%A1%83%E7%8C%BF?tab=pitching) |  
    [Lions](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/Xs1sP-%E7%B5%B1%E4%B8%807-ELEVEn%E7%8D%85?tab=pitching)

- **Cleaning:** unify schema (rename), compute SO/BB (CPBL), drop small-BF pitchers  
- **Merge:** keep common cols + league-only cols (Whiff%, Swing%, PutAway%), include **BAbip**  
- **Imputation:** **KNNImputer for WAR** (features include BAbip) + **IterativeImputer** with team/league encodings  
- **Outputs:** *before-imputation* vs *after-imputation* datasets for comparisons  
    """)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Combined (clean, before imputation) — head:**")
        st.dataframe(combined_clean_no_impute.head())
    with c2:
        st.markdown("**Combined (after imputation) — head:**")
        st.dataframe(combined_imputed.head())

# --------------------------- IDA (Raw) ------------------------------
with tab2:
    st.subheader("Initial Data Analysis (Raw files)")
    colA, colB = st.columns(2)
    with colA:
        st.write("**MLB raw shape:**", mlb_raw.shape)
        st.write("**MLB raw dtypes:**")
        st.dataframe(pd.DataFrame(mlb_raw.dtypes, columns=["dtype"]))
        st.write("**MLB raw missing:**")
        st.dataframe(missing_table(mlb_raw))
    with colB:
        st.write("**CPBL raw shape:**", cpbl_raw.shape)
        st.write("**CPBL raw dtypes:**")
        st.dataframe(pd.DataFrame(cpbl_raw.dtypes, columns=["dtype"]))
        st.write("**CPBL raw missing:**")
        st.dataframe(missing_table(cpbl_raw))

    st.markdown("**Duplicate rows (raw)**")
    dup_mlb = mlb_raw.duplicated().sum()
    dup_cpbl = cpbl_raw.duplicated().sum()
    st.dataframe(pd.DataFrame({
        "MLB_raw_duplicates": [dup_mlb],
        "CPBL_raw_duplicates": [dup_cpbl]
    }))

# --------------------------- EDA (Clean/Imputed) --------------------
with tab3:
    st.subheader("Exploratory Data Analysis")

    st.markdown("**Descriptive statistics (clean before imputation)**")
    desc_num_clean, desc_cat_clean = describe_by_dtype(combined_clean_no_impute)
    st.write("Numeric summary:"); st.dataframe(desc_num_clean)
    st.write("Categorical summary:"); st.dataframe(desc_cat_clean)

    st.markdown("**Descriptive statistics (after imputation)**")
    desc_num_imp, desc_cat_imp = describe_by_dtype(combined_imputed)
    st.write("Numeric summary:"); st.dataframe(desc_num_imp)
    st.write("Categorical summary:"); st.dataframe(desc_cat_imp)

    st.markdown("**Missing values comparison**")
    mt_clean = missing_table(combined_clean_no_impute)
    mt_imp   = missing_table(combined_imputed)
    d1, d2 = st.columns(2)
    with d1:
        st.write("Before imputation:"); st.dataframe(mt_clean)
    with d2:
        st.write("After imputation:"); st.dataframe(mt_imp if len(mt_imp) else pd.DataFrame({"missing":[],"missing_%":[]}))

    # -------- Interactive correlation heatmap (before/after) --------
    if show_heatmap:
        st.markdown("**Interactive correlation heatmap**")
        heatmap_choice = st.radio(
            "Select dataset:", ("Before Imputation", "After Imputation"),
            horizontal=True, key="heatmap_choice"
        )

        df_corr = combined_clean_no_impute.copy() if heatmap_choice == "Before Imputation" else combined_imputed.copy()
        numeric_df = df_corr.select_dtypes(include=["float64", "int64"]).replace([np.inf, -np.inf], np.nan)

        exclude_cols = ["Team_code", "League_code"]
        if exclude_bf_in_heatmap:
            exclude_cols.append("BF")

        candidate_cols = [c for c in numeric_df.columns if c not in exclude_cols]
        valid_corr_cols = [c for c in candidate_cols if numeric_df[c].notna().sum() > 1]

        if len(valid_corr_cols) < 2:
            st.warning(
                "Not enough numeric columns to compute a correlation matrix.\n"
                f"Excluded: {exclude_cols}\n"
                f"Numeric candidates: {candidate_cols}"
            )
        else:
            if show_used_cols:
                with st.expander("Columns used in this heatmap"):
                    st.write(valid_corr_cols)

            # 動態尺寸 + 文字大小 + 不使用容器自動寬度
            n = len(valid_corr_cols)
            cell = 0.85 * heatmap_scale
            fig_w = max(8, cell * n + 2)
            fig_h = max(6, cell * n + 2)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)

            corr = numeric_df[valid_corr_cols].corr(method="pearson")
            sns.heatmap(
                corr,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": heatmap_text},
                ax=ax
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=heatmap_text)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=heatmap_text)
            ax.set_title(f"Correlation Heatmap — {heatmap_choice}", fontsize=heatmap_text + 2)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

# --------------------------- Visualizations -------------------------
with tab4:
    st.subheader("Visualizations")

    # KDE by League
    if show_kde:
        st.markdown("**KDE by League (After Imputation)**")
        plot_cols_all = [c for c in combined_imputed.select_dtypes(include=["float64","int64"]).columns
                         if c not in ["Team_code","League_code","BF"]]
        default_cols = [c for c in ["ERA","ERA_plus","FIP","WHIP","K%","BB%","BAbip"] if c in plot_cols_all]
        selected = st.multiselect("Select variables", plot_cols_all, default=default_cols)

        if selected:
            cols = 3
            rows = int(np.ceil(len(selected)/cols))
            fig = plt.figure(figsize=(5 * cols, 3.6 * rows))
            for i, col in enumerate(selected):
                ax = plt.subplot(rows, cols, i + 1)
                for lg in ["MLB", "CPBL"]:
                    sns.kdeplot(
                        data=combined_imputed[combined_imputed["League"] == lg],
                        x=col, fill=True, alpha=0.3, ax=ax, label=lg
                    )
                ax.set_title(col); ax.legend()
            plt.suptitle("Distribution Comparison by League (After Imputation)", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            st.pyplot(fig)

    # Team Violin
    if show_violin:
        st.markdown("---")
        st.markdown("**Team Violin Plot**")
        plot_df = combined_imputed.dropna(subset=[violin_metric, "Team", "League"]).copy()
        if facet_violin_by_league:
            for lg in ["MLB", "CPBL"]:
                sub = plot_df[plot_df["League"] == lg]
                if sub.empty:
                    continue
                fig = plt.figure(figsize=(max(10, 0.5 * sub["Team"].nunique()), 6))
                sns.violinplot(
                    data=sub, x="Team", y=violin_metric, inner="box",
                    linewidth=0.8, palette="tab20"
                )
                plt.title(f"{violin_metric} Distribution by Team — {lg}")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            fig = plt.figure(figsize=(max(10, 0.5 * plot_df["Team"].nunique()), 6))
            sns.violinplot(
                data=plot_df, x="Team", y=violin_metric, inner="box",
                linewidth=0.8, palette="tab20"
            )
            plt.title(f"{violin_metric} Distribution by Team (All)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

    # ERA+ scatter grid
    if show_scatter:
        st.markdown("---")
        st.markdown("**ERA+ vs Other Metrics (Scatter Grid)**")
        candidates = [
            c for c in combined_imputed.select_dtypes(include=np.number).columns
            if c not in ["ERA_plus", "Team_code", "League_code"]
        ]
        picked = st.multiselect("Pick variables (x-axis)", candidates, default=candidates[:max_scatter])

        if picked:
            n = len(picked); cols = 4; rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
            if rows == 1:
                axes = np.array([axes])
            palette = sns.color_palette("tab20", n_colors=combined_imputed["Team"].nunique())
            for i, xcol in enumerate(picked):
                ax = axes[i // cols, i % cols]
                sns.scatterplot(
                    data=combined_imputed, x=xcol, y="ERA_plus",
                    hue="Team", s=50, alpha=0.7, ax=ax, palette=palette, legend=False
                )
                ax.set_title(xcol)
            # clean empty axes
            for j in range(i + 1, rows * cols):
                fig.delaxes(axes[j // cols, j % cols])
            plt.tight_layout()
            st.pyplot(fig)

# --------------------------- Download -------------------------------
with tab5:
    st.subheader("Download processed CSV (after imputation)")
    buf = io.BytesIO()
    combined_imputed.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "Download combined_pitching_processed.csv",
        buf.getvalue(),
        file_name="combined_pitching_processed.csv",
        mime="text/csv"
    )
