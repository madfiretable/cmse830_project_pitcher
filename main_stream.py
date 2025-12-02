# ============================================================
# test_stream.py  — Pitching Analytics (MLB 2025 + CPBL 2024/2025)
# Version 3.0 (A+B): Multi-year + Year-aware Visualizations
# ============================================================

import io
import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Fix for Arrow serialization issues
def make_arrow_safe(df):
    """Convert dataframe to Arrow-compatible format"""
    df_safe = df.copy()
    for col in df_safe.columns:
        if df_safe[col].dtype == 'object':
            df_safe[col] = df_safe[col].astype(str)
        elif df_safe[col].dtype.name.startswith('int') or df_safe[col].dtype.name.startswith('Int'):
            df_safe[col] = df_safe[col].astype(float)
    return df_safe

# --------------------------- Page/UI config ---------------------------
st.set_page_config(
    page_title="Pitching Analytics — MLB 2025 + CPBL 2024/2025",
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
    "<div class='app-header'><h2 style='margin:0'>⚾ Pitching Analytics (MLB 2025 + CPBL 2024/2025)</h2>"
    "<p style='margin:0.2rem 0 0; opacity:0.8'>Version 3.0 — Multi-Year, Year-aware Visualizations</p></div>",
    unsafe_allow_html=True
)

# --------------------------- Sidebar (basic controls) ---------------------------
st.sidebar.header("Settings")
BF_THRESHOLD = st.sidebar.number_input("BF threshold (drop if BF < threshold)", min_value=0, value=70, step=5)

st.sidebar.subheader("Global Filters")
league_opt = st.sidebar.multiselect("League", ["MLB", "CPBL"], default=["MLB", "CPBL"])
team_opt_placeholder = st.sidebar.empty()

# Year filter（A+B，要支援 2024 + 2025）
st.sidebar.subheader("Year Filter")
all_years_global = [2024, 2025]
year_opt = st.sidebar.multiselect("Select Year", all_years_global, default=all_years_global)

st.sidebar.subheader("EDA Controls")
show_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
exclude_bf_in_heatmap = st.sidebar.checkbox("Exclude BF from Heatmap", value=True)
show_used_cols = st.sidebar.checkbox("Show heatmap variable list", value=False)

st.sidebar.subheader("Heatmap Tuning")
heatmap_scale = st.sidebar.slider("Cell size scale", 0.6, 1.8, 1.2, 0.1)
heatmap_text = st.sidebar.slider("Text size", 6, 18, 11, 1)

st.sidebar.subheader("Visualizations")
show_kde = st.sidebar.checkbox("Show KDE by League/Year (interactive)", value=True)

# Violin controls
show_violin = st.sidebar.checkbox("Show Team Violin Plot (Seaborn)", value=True)
violin_metric = st.sidebar.selectbox(
    "Violin metric (by Team)",
    ["ERA_plus", "ERA", "FIP", "WHIP", "K%", "BB%", "WAR", "HR9", "BAbip"],
    index=0,
)
violin_custom_box = st.sidebar.empty()

show_scatter = st.sidebar.checkbox("Show ERA+ vs Others (Scatter Grid, team-colored, year-shaped)", value=True)
max_scatter = st.sidebar.slider("Max scatter panels", 1, 20, 12)

# --------------------------- Data paths -------------------------------
MLB_PATH = "MLB_Pitch.csv"         # MLB 2025
CPBL_2024_PATH = "投手2024.csv"     # CPBL 2024
CPBL_2025_PATH = "投手.xlsx"        # CPBL 2025

def assert_file(path: str):
    if not os.path.exists(path):
        st.error(f"File not found: `{path}`.")
        st.stop()

assert_file(MLB_PATH)
assert_file(CPBL_2024_PATH)
assert_file(CPBL_2025_PATH)

# --------------------------- Load multi-year CPBL ---------------------------
def load_cpbl_multi(cpbl2024_path: str, cpbl2025_path: str):
    def clean(df, year):
        df = df.copy()
        rename_map = {
            "球員": "Player", "背號": "No", "球隊": "Team",
            "AVG": "BA", "BABIP": "BAbip", "ERA+": "ERA_plus",
            "ERA": "ERA", "FIP": "FIP", "WHIP": "WHIP",
            "OPS": "OPS", "K%": "K%", "BB%": "BB%",
            "Whiff%": "Whiff%", "Swing%": "Swing%",
            "PutAway%": "PutAway%", "GB%": "GB%", "FB%": "FB%",
            "IP": "IP", "BF": "BF"
        }
        df.rename(columns=rename_map, inplace=True)

        if ("K%" in df.columns) and ("BB%" in df.columns):
            df["SO/BB"] = (df["K%"] / df["BB%"]).replace([np.inf, -np.inf], np.nan)

        df["Year"] = year
        return df

    df2024 = clean(pd.read_csv(cpbl2024_path), 2024)
    df2025 = clean(pd.read_excel(cpbl2025_path), 2025)

    return pd.concat([df2024, df2025], ignore_index=True)

# ---- Load MLB & CPBL ----
mlb_raw = pd.read_csv(MLB_PATH)
mlb_raw["Year"] = 2025  # MLB = 2025

cpbl_raw = load_cpbl_multi(CPBL_2024_PATH, CPBL_2025_PATH)

# ============================================================
# PART 2 — Merge & Imputation
# ============================================================

def prepare_and_merge(mlb_df: pd.DataFrame, cpbl_df: pd.DataFrame, bf_threshold: int):
    """Clean MLB + CPBL multi-year datasets and merge into unified schema."""

    cpbl = cpbl_df.copy()
    mlb = mlb_df.copy()

    # ----- CPBL: Compute SO/BB -----
    if ("K%" in cpbl.columns) and ("BB%" in cpbl.columns):
        cpbl["SO/BB"] = (cpbl["K%"] / cpbl["BB%"]).replace([np.inf, -np.inf], np.nan)

    # ----- Standardize column names -----
    cpbl = cpbl.rename(columns={
        "球員": "Player", "背號": "No", "球隊": "Team",
        "AVG": "BA", "BABIP": "BAbip", "ERA+": "ERA_plus",
        "FIP": "FIP", "WHIP": "WHIP", "OPS": "OPS",
        "K%": "K%", "BB%": "BB%", "Whiff%": "Whiff%",
        "Swing%": "Swing%", "PutAway%": "PutAway%",
        "GB%": "GB%", "FB%": "FB%"
    })

    mlb = mlb.rename(columns={"ERA+": "ERA_plus"})
    if "Team" not in mlb.columns:
        mlb["Team"] = mlb.get("Tm", "Unknown")

    # ----- Ensure BF is numeric -----
    for df in (mlb, cpbl):
        df["BF"] = pd.to_numeric(df.get("BF", np.nan), errors="coerce")

    # ----- BF filtering -----
    mlb_before, cpbl_before = len(mlb), len(cpbl)
    mlb_clean = mlb.loc[mlb["BF"].replace([np.inf, -np.inf], np.nan) >= bf_threshold].copy()
    cpbl_clean = cpbl.loc[cpbl["BF"].replace([np.inf, -np.inf], np.nan) >= bf_threshold].copy()
    mlb_after, cpbl_after = len(mlb_clean), len(cpbl_clean)

    # ----- Final schema -----
    common_cols = [
        "Player","Team","IP","BF","ERA","ERA_plus","FIP","WHIP",
        "K%","BB%","GB%","FB%","SO/BB","BAbip","Year"
    ]
    mlb_unique  = ["WAR","HR9"]
    cpbl_unique = ["Whiff%","Swing%","PutAway%"]

    final_cols = common_cols + mlb_unique + cpbl_unique

    # ----- Ensure all required columns exist -----
    for col in final_cols:
        if col not in mlb_clean.columns:
            mlb_clean[col] = np.nan
        if col not in cpbl_clean.columns:
            cpbl_clean[col] = np.nan

    # ----- Add league -----
    mlb_clean["League"]  = "MLB"
    cpbl_clean["League"] = "CPBL"

    # ----- Ensure Year exists for both -----
    if "Year" not in mlb_clean.columns:
        mlb_clean["Year"] = np.nan
    if "Year" not in cpbl_clean.columns:
        cpbl_clean["Year"] = np.nan

    # ----- Keep all final columns -----
    mlb_clean  = mlb_clean[final_cols + ["League"]]
    cpbl_clean = cpbl_clean[final_cols + ["League"]]

    # ----- Merge -----
    combined_clean_no_impute = pd.concat([mlb_clean, cpbl_clean], ignore_index=True)

    artifacts = {
        "counts": {
            "mlb_before": mlb_before, "mlb_after": mlb_after,
            "cpbl_before": cpbl_before, "cpbl_after": cpbl_after
        },
        "mlb_clean": mlb_clean,
        "cpbl_clean": cpbl_clean
    }
    return combined_clean_no_impute, artifacts


def impute_all(combined_clean_no_impute: pd.DataFrame) -> pd.DataFrame:
    combined = combined_clean_no_impute.copy()

    # ----- KNN-impute WAR first -----
    ref_cols = [c for c in [
        "ERA","FIP","WHIP","K%","BB%","SO/BB","HR9","ERA_plus","BAbip"
    ] if c in combined.columns]

    if "WAR" in combined.columns and len(ref_cols) > 0:
        knn = KNNImputer(n_neighbors=8, weights="distance")
        combined["WAR"] = knn.fit_transform(
            combined[ref_cols + ["WAR"]]
        )[:, -1]

    # ----- Encode team + league -----
    combined["Team_code"]   = pd.factorize(combined["Team"])[0].astype(float)
    combined["League_code"] = (combined["League"] == "MLB").astype(float)

    # ----- Select numeric columns -----
    numeric_cols = combined.select_dtypes(include=["float64","int64"]).columns.tolist()
    X = combined[numeric_cols].replace([np.inf, -np.inf], np.nan)

    valid_cols = [c for c in X.columns if X[c].notna().sum() > 0]

    # ----- Scale → Impute → Unscale -----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[valid_cols])

    imp = IterativeImputer(max_iter=30, random_state=42, tol=1e-3)
    X_imp_scaled = imp.fit_transform(X_scaled)
    X_imp = scaler.inverse_transform(X_imp_scaled)

    X_filled = X.copy()
    X_filled[valid_cols] = X_imp
    combined[numeric_cols] = X_filled[numeric_cols]

    # ----- Safety clipping -----
    for c in [c for c in combined.columns if c.endswith("%")]:
        combined[c] = combined[c].clip(0, 100)
    if "HR9" in combined.columns:
        combined["HR9"] = combined["HR9"].clip(lower=0)

    return combined

# ============================================================
# Helper Functions
# ============================================================

def describe_by_dtype(df: pd.DataFrame):
    """Return descriptive stats for numeric and categorical columns.
       Force all categorical outputs to string to avoid Arrow errors."""
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    # Numeric summary
    desc_num = df[num_cols].describe().T if len(num_cols) else pd.DataFrame()

    # Categorical summary
    if len(cat_cols):
        nunique = df[cat_cols].nunique().astype(str).rename("nunique")
        top = df[cat_cols].agg(
            lambda s: str(s.value_counts(dropna=False).index[0]) if s.count() else ""
        ).rename("top")
        freq = df[cat_cols].agg(
            lambda s: str(s.value_counts(dropna=False).iloc[0]) if s.count() else ""
        ).rename("freq")

        desc_cat = pd.concat([nunique, top, freq], axis=1).astype(str)
        desc_cat = make_arrow_safe(desc_cat)
    else:
        desc_cat = pd.DataFrame()

    return desc_num, desc_cat

def missing_table(df: pd.DataFrame):
    """Return missing-value summary sorted by missing percentage."""
    nulls = df.isna().sum()
    pct = (nulls / len(df) * 100).round(2)

    out = pd.DataFrame({"missing": nulls, "missing_%": pct})
    return out[out["missing"] > 0].sort_values("missing_%", ascending=False)

# ============================================================
# PART 3 — Pipeline + Tabs
# ============================================================

combined_clean_no_impute, art = prepare_and_merge(mlb_raw, cpbl_raw, BF_THRESHOLD)
combined_imputed = impute_all(combined_clean_no_impute)

st.sidebar.markdown("---")
st.sidebar.write(
    f"MLB kept: {art['counts']['mlb_after']}/{art['counts']['mlb_before']} | "
    f"CPBL kept: {art['counts']['cpbl_after']}/{art['counts']['cpbl_before']}"
)

all_teams = sorted(combined_imputed["Team"].dropna().unique().tolist())
team_opt = team_opt_placeholder.multiselect("Team", all_teams, default=[])

# Global filter (League + Year + Team)
filtered_df = combined_imputed[
    combined_imputed["League"].isin(league_opt) &
    combined_imputed["Year"].isin(year_opt) &
    (combined_imputed["Team"].isin(team_opt) if team_opt else True)
].copy()

# Violin 自選隊伍（可覆蓋全域 team 選擇）
violin_custom_teams = violin_custom_box.multiselect(
    "Custom teams for violin (single plot when not empty)",
    options=all_teams,
    default=[]
)

# Team color map (for scatter)
qual = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Plotly
)
def make_color_map(categories):
    return {cat: qual[i % len(qual)] for i, cat in enumerate(categories)}
team_color_map = make_color_map(sorted(filtered_df["Team"].dropna().unique()))

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "IDA (Raw)", "EDA (Clean/Imputed)", "Visualizations", "Download"
])

# --------------------------- Overview ---------------------------
with tab1:
    st.subheader("Data Sources and Pipeline")
    st.markdown("""
    ### **Datasets**
    - **MLB 2025** — MLB_Pitch.csv  
    - **CPBL 2024** — 投手2024.csv  
    - **CPBL 2025** — 投手.xlsx  

    ### **Process**
    1. Load MLB (Year = 2025)  
    2. Load CPBL 2024 & 2025 → combine  
    3. Unify schema  
    4. Compute SO/BB (CPBL)  
    5. BF threshold  
    6. Imputation (KNN → Iterative Imputer)  
    
    ### **Terms Explanation**
    """)
    
    # Create metrics explanation dataframe
    metrics_data = {
        "Metric": [
            "BF", "ERA", "ERA+", "FIP", "WHIP", "K%", "BB%", "GB%", "FB%", 
            "SO/BB", "BABIP", "WAR", "HR/9", "Whiff%", "Swing%", "PutAway%"
        ],
        "Meaning / Definition": [
            "Batters Faced — total number of hitters a pitcher has faced.",
            "Earned Run Average — earned runs allowed per 9 innings.",
            "Adjusted ERA — normalizes ERA for league and ballpark factors (100 = league average).",
            "Fielding Independent Pitching — focuses on outcomes under pitcher's control (K, BB, HR, HBP).",
            "Walks + Hits per Inning Pitched.",
            "Strikeout Percentage — strikeouts ÷ batters faced × 100.",
            "Walk Percentage — walks ÷ batters faced × 100.",
            "Ground Ball Percentage — share of batted balls hit on the ground.",
            "Fly Ball Percentage — share of batted balls hit in the air.",
            "Strikeout-to-Walk Ratio — strikeouts ÷ walks.",
            "Batting Average on Balls In Play — excludes HR and strikeouts.",
            "Wins Above Replacement — total value above a replacement-level player.",
            "Home Runs per 9 Innings — HR allowed × 9 ÷ innings pitched.",
            "Swinging Strike Rate — % of swings that miss completely.",
            "Swing Rate — % of total pitches that batters swing at.",
            "Putaway Rate — % of two-strike counts ending in strikeout."
        ],
        "Interpretation": [
            "Measures workload; higher means more innings pitched.",
            "Lower ERA = better run prevention.",
            ">100 = above average, <100 = below average.",
            "Lower FIP = better true pitching skill.",
            "Lower WHIP = fewer baserunners, better control.",
            "Higher K% = more dominant pitching.",
            "Lower BB% = better control and command.",
            "Higher GB% = induces weak contact, fewer HRs.",
            "High FB% = more flyouts but greater HR risk.",
            "Higher ratio = efficient, dominant pitcher.",
            "~.300 is typical; much higher/lower may suggest luck.",
            "Higher WAR = greater overall contribution.",
            "Lower HR/9 = better at limiting long balls.",
            "High Whiff% = strong pitch movement/deception.",
            "Shows how aggressive hitters are against the pitcher.",
            "High PutAway% = finishes hitters efficiently."
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(make_arrow_safe(metrics_df), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Combined (Before Imputation)")
        st.dataframe(make_arrow_safe(combined_clean_no_impute.head()), use_container_width=True)
    with c2:
        st.markdown("### Combined (After Imputation)")
        st.dataframe(make_arrow_safe(combined_imputed.head()), use_container_width=True)

# --------------------------- IDA (Raw) ---------------------------
with tab2:
    st.subheader("Initial Data Analysis (Raw files)")

    colA, colB = st.columns(2)
    with colA:
        st.write("### MLB Raw Shape:", mlb_raw.shape)
        st.dataframe(make_arrow_safe(pd.DataFrame(mlb_raw.dtypes, columns=["dtype"])), use_container_width=True)
        st.write("### MLB Raw Missing")
        st.dataframe(make_arrow_safe(missing_table(mlb_raw)), use_container_width=True)

    with colB:
        st.write("### CPBL Raw Shape:", cpbl_raw.shape)
        st.dataframe(make_arrow_safe(pd.DataFrame(cpbl_raw.dtypes, columns=["dtype"])), use_container_width=True)
        st.write("### CPBL Raw Missing")
        st.dataframe(make_arrow_safe(missing_table(cpbl_raw)), use_container_width=True)

    st.write("### Duplicate Rows Count")
    dup_df = pd.DataFrame({
        "MLB_duplicates": [mlb_raw.duplicated().sum()],
        "CPBL_duplicates": [cpbl_raw.duplicated().sum()],
    })
    st.dataframe(make_arrow_safe(dup_df), use_container_width=True)

# --------------------------- EDA (Clean/Imputed) ---------------------------
with tab3:
    st.subheader("Exploratory Data Analysis")

    # Before Imputation
    st.markdown("### Descriptive Statistics (Clean before Imputation)")
    desc_num_clean, desc_cat_clean = describe_by_dtype(combined_clean_no_impute)
    st.write("#### Numeric Summary")
    st.dataframe(make_arrow_safe(desc_num_clean), use_container_width=True)
    st.write("#### Categorical Summary")
    st.dataframe(make_arrow_safe(desc_cat_clean), use_container_width=True)

    # After Imputation
    st.markdown("### Descriptive Statistics (After Imputation)")
    desc_num_imp, desc_cat_imp = describe_by_dtype(combined_imputed)
    st.write("#### Numeric Summary")
    st.dataframe(make_arrow_safe(desc_num_imp), use_container_width=True)
    st.write("#### Categorical Summary")
    st.dataframe(make_arrow_safe(desc_cat_imp), use_container_width=True)

    # Missing
    st.markdown("### Missing Value Comparison")
    d1, d2 = st.columns(2)
    with d1:
        st.write("Before Imputation")
        st.dataframe(make_arrow_safe(missing_table(combined_clean_no_impute)), use_container_width=True)
    with d2:
        mt_imp = missing_table(combined_imputed)
        st.write("After Imputation")
        st.dataframe(make_arrow_safe(mt_imp if len(mt_imp) else pd.DataFrame({"missing":[],"missing_%":[]})), use_container_width=True)

    # Heatmap
    if show_heatmap:
        st.markdown("### Correlation Heatmap (Interactive)")
        pick = st.radio("Select dataset:", ("Before Imputation", "After Imputation"), horizontal=True, key="hm_pick")

        base_df = combined_clean_no_impute if pick == "Before Imputation" else combined_imputed
        num_df = base_df.select_dtypes(include=["float64", "int64"]).replace([np.inf, -np.inf], np.nan)

        exclude_cols = ["Team_code", "League_code"]
        if exclude_bf_in_heatmap:
            exclude_cols.append("BF")

        cols = [c for c in num_df.columns if c not in exclude_cols and num_df[c].notna().sum() > 1]
        if len(cols) < 2:
            st.info("Not enough numeric columns for heatmap.")
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

# ============================================================
# PART 4 — Visualizations + Models
# ============================================================

with tab4:
    st.subheader("Visualizations")

    # --------------------------- KDE ---------------------------
    if show_kde:
        st.markdown("### KDE by League/Year (Interactive)")

        plot_cols_all = [
            c for c in filtered_df.select_dtypes(include=["float64", "int64"]).columns
            if c not in ["Team_code", "League_code", "BF"]
        ]
        default_cols = [
            c for c in ["ERA", "ERA_plus", "FIP", "WHIP", "K%", "BB%", "BAbip"]
            if c in plot_cols_all
        ]

        selected = st.multiselect(
            "Select variables", plot_cols_all, default=default_cols, key="kde_vars"
        )

        for col in selected:
            hist_data = []
            labels = []

            # Group by (League, Year) — label like "MLB 25", "CPBL 24"
            for (lg, yr), group in filtered_df.groupby(["League", "Year"]):
                vals = group[col].dropna().values
                if len(vals) > 1:
                    yy = int(yr) % 100
                    label = f"{lg} {yy:02d}"
                    hist_data.append(vals)
                    labels.append(label)

            if hist_data:
                fig = ff.create_distplot(
                    hist_data, labels,
                    show_hist=False, show_rug=False
                )
                fig.update_layout(
                    title=f"Density (KDE): {col} by League/Year",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

    # --------------------------- Violin ---------------------------
    if show_violin:
        st.markdown("---")
        sns.set_theme(style="whitegrid")

        chosen_teams = violin_custom_teams if len(violin_custom_teams) > 0 else team_opt

        # Helper: build Team_YY label
        def add_team_yy(df: pd.DataFrame):
            df = df.copy()
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            yy = df["Year"].fillna(0).astype(int) % 100
            df["Team_YY"] = df["Team"].astype(str) + " " + yy.astype(str).str.zfill(2)
            return df

        if chosen_teams:
            sub = combined_imputed.copy()
            sub = sub[sub["League"].isin(league_opt)]
            sub = sub[sub["Year"].isin(year_opt)]
            sub = sub[sub["Team"].isin(chosen_teams)]
            sub = sub[sub[violin_metric].notna()]

            sub = add_team_yy(sub)

            st.markdown(f"### Team Violin Plot — {violin_metric} (Custom Teams, Team YY)")

            if sub.empty:
                st.info("No data to plot. Please select teams/leagues/years.")
            else:
                fig, ax = plt.subplots(
                    figsize=(max(10, 0.55 * sub["Team_YY"].nunique()), 6),
                    dpi=140
                )
                sns.violinplot(
                    data=sub, x="Team_YY", y=violin_metric,
                    inner="box", linewidth=0.8,
                    palette="tab20", hue="Team_YY", legend=False, ax=ax
                )
                ax.set_title(f"{violin_metric} Distribution — Selected Teams (Team YY)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

        else:
            st.markdown(f"### Team Violin Plot — {violin_metric} (Split by League, Team YY)")

            for lg in ["MLB", "CPBL"]:
                sub = filtered_df[(filtered_df["League"] == lg) & filtered_df[violin_metric].notna()]
                if sub.empty:
                    continue

                sub = add_team_yy(sub)

                fig, ax = plt.subplots(
                    figsize=(max(10, 0.55 * sub["Team_YY"].nunique()), 6),
                    dpi=140
                )
                sns.violinplot(
                    data=sub, x="Team_YY", y=violin_metric,
                    inner="box", linewidth=0.8,
                    palette="tab20", hue="Team_YY", legend=False, ax=ax
                )
                ax.set_title(f"{violin_metric} Distribution — {lg} (Team YY)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

    # --------------------------- Scatter ---------------------------
    if show_scatter:
        st.markdown("---")
        st.markdown("### ERA+ vs Other Metrics (Scatter Grid, color=Team, shape=Year)")

        # --- Build clean numeric candidates ---
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        remove_cols = ["ERA_plus", "Team_code", "League_code", "Year"]

        clean_numeric_cols = []
        for col in numeric_cols:
            if col in remove_cols:
                continue
            # 必須保證欄位是一維，不能是 nested array/object
            if filtered_df[col].ndim == 1:
                clean_numeric_cols.append(col)

        candidates = clean_numeric_cols

        picked = st.multiselect(
            "Pick variables (x-axis)",
            candidates,
            default=candidates[:max_scatter],
            key="scatter_vars"
        )

        year_symbols = {2024: "circle", 2025: "square"}

        if picked:
            cols = 4
            rows = int(np.ceil(len(picked) / cols))
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=picked)

            for i, xcol in enumerate(picked):
                r = i // cols + 1
                c = i % cols + 1

                # Skip invalid columns (extra safety)
                if xcol not in filtered_df.columns:
                    continue
                if filtered_df[xcol].ndim != 1:
                    continue

                # Build sub df
                sub = filtered_df[[xcol, "ERA_plus", "Player", "Team", "Year"]].copy()
                sub = sub.dropna(subset=[xcol, "ERA_plus", "Year"])

                # --- Clean Year (force 1D array → numeric → int) ---
                year_raw = sub["Year"].values  # flatten
                sub["Year"] = pd.Series(pd.to_numeric(year_raw, errors="coerce")).fillna(0).astype(int).values

                symbols = [year_symbols.get(y, "circle") for y in sub["Year"]]

                fig.add_trace(
                    go.Scattergl(
                        x=sub[xcol],
                        y=sub["ERA_plus"],
                        mode="markers",
                        marker=dict(
                            size=7,
                            opacity=0.85,
                            color=sub["Team"].map(team_color_map),
                            symbol=symbols,
                        ),
                        customdata=sub[["Player", "Team", "Year"]],
                        hovertemplate=(
                            "Player: %{customdata[0]}<br>"
                            "Team: %{customdata[1]}<br>"
                            "Year: %{customdata[2]}<br>"
                            f"{xcol}: %{{x:.2f}}<br>"
                            "ERA+: %{{y:.2f}}<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=r, col=c,
                )

                fig.update_xaxes(title_text=xcol, row=r, col=c)
                fig.update_yaxes(title_text="ERA_plus", row=r, col=c)

            fig.update_layout(
                height=320 * rows,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)

        # --------------------------- WAR Models ---------------------------
        st.markdown("---")
        st.subheader("WAR Prediction Models")

        features = [
            "ERA", "ERA_plus", "FIP", "WHIP",
            "K%", "BB%", "SO/BB",
            "HR9", "BAbip", "GB%", "FB%"
        ]
        features = [c for c in features if c in combined_imputed.columns]

        df_model = combined_imputed.dropna(subset=["WAR"]).copy()
        X = df_model[features]
        y = df_model["WAR"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # Model A: Linear Regression
        st.markdown("### Model A: Linear Regression")

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        st.write("**R² score:**", round(r2_score(y_test, pred_lr), 4))
        st.write("**MAE:**", round(mean_absolute_error(y_test, pred_lr), 4))

        coef_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": lr.coef_,
        }).sort_values("Coefficient", ascending=False)

        st.write("**Feature Coefficients:**")
        st.dataframe(make_arrow_safe(coef_df), use_container_width=True)

        # Model B: Random Forest
        st.markdown("### Model B: Random Forest Regression")

        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=42
        )
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)

        st.write("**R² score:**", round(r2_score(y_test, pred_rf), 4))
        st.write("**MAE:**", round(mean_absolute_error(y_test, pred_rf), 4))

        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.write("**Feature Importances:**")
        st.dataframe(make_arrow_safe(fi_df), use_container_width=True)

# ============================================================
# PART 5 — Download
# ============================================================

with tab5:
    st.subheader("Download processed CSV (after imputation & filters)")

    buf = io.BytesIO()
    filtered_df.to_csv(buf, index=False, encoding="utf-8-sig")

    st.download_button(
        label="Download CSV",
        data=buf.getvalue(),
        file_name="combined_pitching_processed_filtered.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center; opacity:0.6'>"
    "Pitching Analytics (MLB + CPBL 2024/2025) — Streamlit Dashboard"
    "</p>",
    unsafe_allow_html=True
)
