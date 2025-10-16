# main_stream_pro_v2_violin_seaborn_custom_v2.py
# 變更重點：
# - 小提琴圖：只要偵測到「自選隊伍」（來源可為全域 Team 篩選或本區自選清單），
#   一律把所有自選隊伍畫在「同一張圖」（不分聯盟）。
# - 若沒有任何自選隊伍，才採用「分聯盟」各一張的呈現。
# 其他維持：Heatmap(Plotly互動), KDE(互動), Scatter(隊伍上色), 主題/側欄。

import io
import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------- Page/UI config ---------------------------
st.set_page_config(
    page_title="Pitching Analytics (MLB + CPBL) — Pro v2 (Seaborn Violin - Auto Combine)",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1400px; padding-top: 0.5rem; padding-bottom: 2rem; }
    .app-header {
        padding: 0.8rem 1.2rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.65);
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        backdrop-filter: blur(6px);
        margin-bottom: 0.6rem;
    }
    .metric-card {
        padding: 0.8rem 1rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.65);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div class='app-header'><h2 style='margin:0'>⚾ Pitching Analytics (MLB + CPBL)</h2>"
    "<p style='margin:0.2rem 0 0; opacity:0.8'>Version 2.4: Changed theme and added interactive plots</p></div>",
    unsafe_allow_html=True
)

# --------------------------- Sidebar ---------------------------
st.sidebar.header("Settings")
BF_THRESHOLD = st.sidebar.number_input("BF threshold (drop if BF < threshold)", min_value=0, value=70, step=5)

st.sidebar.subheader("Global Filters")
league_opt = st.sidebar.multiselect("League", ["MLB", "CPBL"], default=["MLB", "CPBL"])
team_opt_placeholder = st.sidebar.empty()

st.sidebar.subheader("EDA Controls")
show_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
exclude_bf_in_heatmap = st.sidebar.checkbox("Exclude BF from Heatmap", value=True)
show_used_cols = st.sidebar.checkbox("Show heatmap variable list", value=False)

st.sidebar.subheader("Heatmap Tuning")
heatmap_scale = st.sidebar.slider("Cell size scale", 0.6, 1.8, 1.2, 0.1)
heatmap_text = st.sidebar.slider("Text size", 6, 18, 11, 1)

st.sidebar.subheader("Visualizations")
show_kde = st.sidebar.checkbox("Show KDE by League (interactive)", value=True)

# Violin controls
show_violin = st.sidebar.checkbox("Show Team Violin Plot (Seaborn)", value=True)
violin_metric = st.sidebar.selectbox(
    "Violin metric (by Team)",
    ["ERA_plus", "ERA", "FIP", "WHIP", "K%", "BB%", "WAR", "HR9", "BAbip"],
    index=0,
)
# 額外提供本區的自選隊伍清單（可與全域 Team 篩選並存）
violin_custom_box = st.sidebar.empty()

show_scatter = st.sidebar.checkbox("Show ERA+ vs Others (Scatter Grid, team-colored)", value=True)
max_scatter = st.sidebar.slider("Max scatter panels", 1, 20, 12)

# --------------------------- Data paths -------------------------------
MLB_PATH = "MLB_Pitch.csv"
CPBL_PATH = "投手.xlsx"

def assert_file(path: str):
    if not os.path.exists(path):
        st.error(f"File not found: `{path}`. Put it in the same folder as this app.")
        st.stop()

assert_file(MLB_PATH)
assert_file(CPBL_PATH)

# --------------------------- Load data ---------------------------
mlb_raw = pd.read_csv(MLB_PATH)
cpbl_raw = pd.read_excel(CPBL_PATH)

# --------------------------- Helpers ---------------------------
def prepare_and_merge(mlb_df: pd.DataFrame, cpbl_df: pd.DataFrame, bf_threshold: int):
    cpbl = cpbl_df.copy()
    mlb = mlb_df.copy()

    if ("K%" in cpbl.columns) and ("BB%" in cpbl.columns):
        cpbl["SO/BB"] = (cpbl["K%"] / cpbl["BB%"]).replace([np.inf, -np.inf], np.nan)

    cpbl = cpbl.rename(columns={
        "球員": "Player", "背號": "No", "球隊": "Team",
        "AVG": "BA", "BABIP": "BAbip", "ERA+": "ERA_plus",
        "FIP": "FIP", "WHIP": "WHIP", "OPS": "OPS",
        "K%": "K%", "BB%": "BB%", "Whiff%": "Whiff%",
        "Swing%": "Swing%", "PutAway%": "PutAway%", "GB%": "GB%", "FB%": "FB%",
    })
    mlb = mlb.rename(columns={"ERA+": "ERA_plus"})
    if "Team" not in mlb.columns:
        mlb["Team"] = mlb.get("Tm", "Unknown")

    for df in (mlb, cpbl):
        df["BF"] = pd.to_numeric(df.get("BF", np.nan), errors="coerce")

    mlb_before, cpbl_before = len(mlb), len(cpbl)
    mlb_clean = mlb.loc[mlb["BF"].replace([np.inf, -np.inf], np.nan) >= bf_threshold].copy()
    cpbl_clean = cpbl.loc[cpbl["BF"].replace([np.inf, -np.inf], np.nan) >= bf_threshold].copy()
    mlb_after, cpbl_after = len(mlb_clean), len(cpbl_clean)

    common_cols = ["Player","Team","IP","BF","ERA","ERA_plus","FIP","WHIP","K%","BB%","GB%","FB%","SO/BB","BAbip"]
    mlb_unique  = ["WAR","HR9"]
    cpbl_unique = ["Whiff%","Swing%","PutAway%"]
    final_cols  = common_cols + mlb_unique + cpbl_unique

    for col in final_cols:
        if col not in mlb_clean.columns:  mlb_clean[col]  = np.nan
        if col not in cpbl_clean.columns: cpbl_clean[col] = np.nan

    mlb_clean["League"]  = "MLB"
    cpbl_clean["League"] = "CPBL"
    mlb_clean  = mlb_clean[final_cols + ["League"]]
    cpbl_clean = cpbl_clean[final_cols + ["League"]]

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
    combined = combined_clean_no_impute.copy()

    ref_cols = [c for c in ["ERA","FIP","WHIP","K%","BB%","SO/BB","HR9","ERA_plus","BAbip"] if c in combined.columns]
    if "WAR" in combined.columns and len(ref_cols) > 0:
        knn = KNNImputer(n_neighbors=8, weights="distance")
        combined["WAR"] = knn.fit_transform(combined[ref_cols + ["WAR"]])[:, -1]

    combined["Team_code"]   = pd.factorize(combined["Team"])[0].astype(float)
    combined["League_code"] = (combined["League"] == "MLB").astype(float)

    numeric_cols = combined.select_dtypes(include=["float64","int64"]).columns.tolist()
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

    for c in [c for c in combined.columns if c.endswith("%")]:
        combined[c] = combined[c].clip(0, 100)
    if "HR9" in combined.columns:
        combined["HR9"] = combined["HR9"].clip(lower=0)
    return combined

def describe_by_dtype(df: pd.DataFrame):
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

# --------------------------- Pipeline ---------------------------
combined_clean_no_impute, art = prepare_and_merge(mlb_raw, cpbl_raw, BF_THRESHOLD)
combined_imputed = impute_all(combined_clean_no_impute)

st.sidebar.markdown("---")
st.sidebar.write(
    f"MLB kept: {art['counts']['mlb_after']}/{art['counts']['mlb_before']} | "
    f"CPBL kept: {art['counts']['cpbl_after']}/{art['counts']['cpbl_before']}"
)
all_teams = sorted(combined_imputed["Team"].dropna().unique().tolist())
team_opt = team_opt_placeholder.multiselect("Team", all_teams, default=[])

# 套用全域篩選（給非 Violin 圖用）
filtered_df = combined_imputed[
    combined_imputed["League"].isin(league_opt) &
    (combined_imputed["Team"].isin(team_opt) if team_opt else True)
].copy()

# Violin 專用：提供本區的自選清單（可覆蓋全域 Team 篩選）
violin_custom_teams = violin_custom_box.multiselect(
    "Custom teams for violin (single plot when not empty)",
    options=all_teams,
    default=[]
)

# Team colors (for scatter)
qual = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Plotly
)
def make_color_map(categories):
    return {cat: qual[i % len(qual)] for i, cat in enumerate(categories)}
team_color_map = make_color_map(sorted(filtered_df["Team"].dropna().unique()))

# --------------------------- Tabs ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "IDA (Raw)", "EDA (Clean/Imputed)", "Visualizations", "Download"
])

# --------------------------- Overview ---------------------------
with tab1:
    st.subheader("Data sources and pipeline")
    st.markdown("""
- **Sources:** 
  - **MLB:**  
    [Standard Pitching (MLB)](https://www.baseball-reference.com/leagues/majors/2025-standard-pitching.shtml),  
    [Advanced Pitching (MLB)](https://www.baseball-reference.com/leagues/majors/2025-advanced-pitching.shtml)  
  - **CPBL (Chinese Professional Baseball League):**  
    [Brothers](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/Kae1X-%E4%B8%AD%E4%BF%A1%E5%85%84%E5%BC%9F?tab=pitching) |  
    [Hawks](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/t6zJf-%E5%8F%B0%E9%8B%BC%E9%9B%84%E9%B7%B9?tab=pitching) |  
    [Dragons](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/R2VRh-%E5%91%B3%E5%85%A8%E9%BE%8D?tab=pitching) |  
    [Guardians](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/wi4T3-%E5%AF%8C%E9%82%A6%E6%82%8D%E5%B0%87?tab=pitching) |  
    [Monkeys](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/WyADE-%E6%A8%82%E5%A4%A9%E6%A1%83%E7%8C%BF?tab=pitching) |  
    [Lions](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/Xs1sP-%E7%B5%B1%E4%B8%807-ELEVEn%E7%8D%85?tab=pitching)
- **Cleaning:** unify schema, compute SO/BB (CPBL), BF filter  
- **Merge:** common cols + league-only cols, include **BAbip**  
- **Imputation:** KNNImputer(WAR) + IterativeImputer(team/league-aware)
    """)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Combined (clean, before imputation) — head:**")
        st.dataframe(combined_clean_no_impute.head(), use_container_width=True)
    with c2:
        st.markdown("**Combined (after imputation) — head:**")
        st.dataframe(combined_imputed.head(), use_container_width=True)

# --------------------------- IDA (Raw) ---------------------------
with tab2:
    st.subheader("Initial Data Analysis (Raw files)")
    colA, colB = st.columns(2)
    with colA:
        st.write("**MLB raw shape:**", mlb_raw.shape)
        st.write("**MLB raw dtypes:**")
        st.dataframe(pd.DataFrame(mlb_raw.dtypes, columns=["dtype"]), use_container_width=True)
        st.write("**MLB raw missing:**")
        st.dataframe(missing_table(mlb_raw), use_container_width=True)
    with colB:
        st.write("**CPBL raw shape:**", cpbl_raw.shape)
        st.write("**CPBL raw dtypes:**")
        st.dataframe(pd.DataFrame(cpbl_raw.dtypes, columns=["dtype"]), use_container_width=True)
        st.write("**CPBL raw missing:**")
        st.dataframe(missing_table(cpbl_raw), use_container_width=True)

    st.markdown("**Duplicate rows (raw)**")
    dup_mlb = mlb_raw.duplicated().sum()
    dup_cpbl = cpbl_raw.duplicated().sum()
    st.dataframe(pd.DataFrame({
        "MLB_raw_duplicates": [dup_mlb],
        "CPBL_raw_duplicates": [dup_cpbl]
    }), use_container_width=True)

# --------------------------- EDA (Clean/Imputed) --------------------
with tab3:
    st.subheader("Exploratory Data Analysis")

    st.markdown("**Descriptive statistics (clean before imputation)**")
    desc_num_clean, desc_cat_clean = describe_by_dtype(combined_clean_no_impute)
    st.write("Numeric summary:"); st.dataframe(desc_num_clean, use_container_width=True)
    st.write("Categorical summary:"); st.dataframe(desc_cat_clean, use_container_width=True)

    st.markdown("**Descriptive statistics (after imputation)**")
    desc_num_imp, desc_cat_imp = describe_by_dtype(combined_imputed)
    st.write("Numeric summary:"); st.dataframe(desc_num_imp, use_container_width=True)
    st.write("Categorical summary:"); st.dataframe(desc_cat_imp, use_container_width=True)

    st.markdown("**Missing values comparison**")
    d1, d2 = st.columns(2)
    with d1:
        st.write("Before imputation:"); st.dataframe(missing_table(combined_clean_no_impute), use_container_width=True)
    with d2:
        mt_imp = missing_table(combined_imputed)
        st.write("After imputation:"); st.dataframe(mt_imp if len(mt_imp) else pd.DataFrame({"missing":[],"missing_%":[]}), use_container_width=True)

    # Heatmap: interactive Plotly
    if show_heatmap:
        st.markdown("**Correlation heatmap (interactive)**")
        pick = st.radio("Select dataset:", ("Before Imputation", "After Imputation"), horizontal=True, key="hm_pick")

        base_df = combined_clean_no_impute if pick == "Before Imputation" else combined_imputed
        num_df = base_df.select_dtypes(include=["float64", "int64"]).replace([np.inf, -np.inf], np.nan)

        exclude_cols = ["Team_code", "League_code"]
        if exclude_bf_in_heatmap:
            exclude_cols.append("BF")

        cols = [c for c in num_df.columns if c not in exclude_cols and num_df[c].notna().sum() > 1]
        if len(cols) < 2:
            st.info("Not enough numeric columns for correlation heatmap.")
        else:
            corr = num_df[cols].corr().round(2)
            annotated = ff.create_annotated_heatmap(
                z=corr.values, x=list(corr.columns), y=list(corr.index),
                colorscale="RdBu", zmin=-1, zmax=1, showscale=True, hoverinfo="z"
            )
            for ann in annotated.layout.annotations:
                ann.font.size = heatmap_text
            n = len(cols)
            height = min(1000, int(80 + (28 * heatmap_scale) * n))
            annotated.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(annotated, use_container_width=True)

# --------------------------- Visualizations -------------------------
with tab4:
    st.subheader("Visualizations")

    # KDE（Plotly）
    if show_kde:
        st.markdown("### KDE by League (Interactive)")
        plot_cols_all = [c for c in filtered_df.select_dtypes(include=["float64","int64"]).columns
                         if c not in ["Team_code","League_code","BF"]]
        default_cols = [c for c in ["ERA","ERA_plus","FIP","WHIP","K%","BB%","BAbip"] if c in plot_cols_all]
        selected = st.multiselect("Select variables", plot_cols_all, default=default_cols, key="kde_vars")
        for col in selected:
            series_mlb  = filtered_df.loc[filtered_df["League"]=="MLB", col].dropna().values
            series_cpbl = filtered_df.loc[filtered_df["League"]=="CPBL", col].dropna().values
            hist_data, labels = [], []
            if len(series_mlb)  > 1: hist_data.append(series_mlb);  labels.append("MLB")
            if len(series_cpbl) > 1: hist_data.append(series_cpbl); labels.append("CPBL")
            if hist_data:
                fig = ff.create_distplot(hist_data, labels, show_hist=False, show_rug=False)
                fig.update_layout(title=f"Density (KDE): {col} by League", margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

    # ---- Violin（Seaborn）：有自選就單圖、沒自選才分聯盟 ----
    if show_violin:
        st.markdown("---")
        sns.set_theme(style="whitegrid")

        # 決定「自選隊伍集合」：優先使用 Violin 區塊自選；若空，再用全域 Team 篩選
        chosen_teams = violin_custom_teams if len(violin_custom_teams) > 0 else team_opt

        if chosen_teams:
            # 單張圖，不分聯盟：只取被選隊伍，且尊重「聯盟全域篩選」
            sub = combined_imputed.copy()
            sub = sub[sub["League"].isin(league_opt)]
            sub = sub[sub["Team"].isin(chosen_teams)]
            sub = sub[sub[violin_metric].notna()]

            st.markdown(f"### Team Violin Plot — {violin_metric}")
            if sub.empty:
                st.info("No data to plot. Please make sure you choosed a team or a league。")
            else:
                fig, ax = plt.subplots(figsize=(max(10, 0.55 * sub['Team'].nunique()), 6), dpi=140)
                sns.violinplot(
                    data=sub, x="Team", y=violin_metric,
                    inner="box", linewidth=0.8, palette="tab20", ax=ax
                )
                ax.set_title(f"{violin_metric} Distribution by Team — Custom Selection (Merged)")
                ax.set_xlabel("Team"); ax.set_ylabel(violin_metric)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

        else:
            # 沒有任何自選隊伍 → 分聯盟各一張
            st.markdown(f"### Team Violin Plot — {violin_metric}")
            for lg in ["MLB", "CPBL"]:
                sub = filtered_df[(filtered_df["League"] == lg) & filtered_df[violin_metric].notna()]
                if sub.empty: 
                    continue
                fig, ax = plt.subplots(figsize=(max(10, 0.55 * sub['Team'].nunique()), 6), dpi=140)
                sns.violinplot(
                    data=sub, x="Team", y=violin_metric,
                    inner="box", linewidth=0.8, palette="tab20", ax=ax
                )
                ax.set_title(f"{violin_metric} Distribution by Team — {lg}")
                ax.set_xlabel("Team"); ax.set_ylabel(violin_metric)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

    # ERA+ Scatter Grid（Plotly、依隊伍上色）
    if show_scatter:
        st.markdown("---")
        st.markdown("### ERA+ vs Other Metrics (Scatter Grid, colored by Team)")
        candidates = [
            c for c in filtered_df.select_dtypes(include=np.number).columns
            if c not in ["ERA_plus","Team_code","League_code"]
        ]
        picked = st.multiselect("Pick variables (x-axis)", candidates, default=candidates[:max_scatter], key="scatter_vars")

        if picked:
            cols = 4
            rows = int(np.ceil(len(picked) / cols))
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=picked)
            for i, xcol in enumerate(picked):
                r = i // cols + 1
                c = i % cols + 1
                sub = filtered_df[[xcol, "ERA_plus", "Player", "Team"]].dropna()
                color_list = sub["Team"].map(team_color_map).tolist()
                fig.add_trace(
                    go.Scattergl(
                        x=sub[xcol], y=sub["ERA_plus"],
                        mode="markers",
                        marker=dict(size=7, opacity=0.8, color=color_list),
                        customdata=sub[["Player","Team"]],
                        hovertemplate=(
                            "Player: %{customdata[0]}<br>"
                            "Team: %{customdata[1]}<br>"
                            f"{xcol}: %{{x:.2f}}<br>"
                            "ERA+: %{{y:.2f}}<extra></extra>"
                        ),
                        showlegend=False
                    ),
                    row=r, col=c
                )
                fig.update_xaxes(title_text=xcol, row=r, col=c)
                fig.update_yaxes(title_text="ERA_plus", row=r, col=c)

            fig.update_layout(height=320*rows, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

# --------------------------- Download -------------------------------
with tab5:
    st.subheader("Download processed CSV (after imputation & filters)")
    buf = io.BytesIO()
    filtered_df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "Download CSV",
        buf.getvalue(),
        file_name="combined_pitching_processed_filtered.csv",
        mime="text/csv"
    )

