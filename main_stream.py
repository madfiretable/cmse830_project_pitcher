# app.py — 本機讀檔、不用上傳；已加入 BAbip
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

# ---------------- Page & style ----------------
st.set_page_config(page_title="Pitching Analytics (MLB + CPBL)", layout="wide")
st.title("Pitching Analytics (MLB + CPBL) – Local Files")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")
BF_THRESHOLD = st.sidebar.number_input("BF threshold (drop if BF < threshold)",
                                       min_value=0, value=70, step=5)
show_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
exclude_bf_in_heatmap = st.sidebar.checkbox("Exclude BF from Heatmap", value=True)
show_kde = st.sidebar.checkbox("Show KDE by League", value=True)
show_violin = st.sidebar.checkbox("Show Team Violin Plot", value=True)
violin_metric = st.sidebar.selectbox(
    "Violin metric (by Team)",
    ["ERA_plus", "ERA", "FIP", "WHIP", "K%", "BB%", "WAR", "HR9", "BAbip"],  # + BAbip
    index=0
)
facet_violin_by_league = st.sidebar.checkbox("Facet Violin by League", value=True)
show_scatter = st.sidebar.checkbox("Show ERA+ vs Others (Scatter Grid)", value=True)
max_scatter = st.sidebar.slider("Max scatter panels", 1, 20, 12)

# ---------------- Load local files ----------------
MLB_PATH = "MLB_Pitch.csv"
CPBL_PATH = "投手.xlsx"

def assert_file(path: str):
    if not os.path.exists(path):
        st.error(f"File not found: `{path}`. Put it in the same folder as this app.")
        st.stop()

assert_file(MLB_PATH)
assert_file(CPBL_PATH)

mlb = pd.read_csv(MLB_PATH)
cpbl = pd.read_excel(CPBL_PATH)

# ---------------- SAME pipeline (with BAbip) ----------------
# 1) Create SO/BB in CPBL (avoid division by zero)
if ("K%" in cpbl.columns) and ("BB%" in cpbl.columns):
    cpbl["SO/BB"] = (cpbl["K%"] / cpbl["BB%"]).replace([np.inf, -np.inf], np.nan)

# 2) Rename CPBL columns to align with MLB
rename_map_cpbl = {
    "球員": "Player",
    "背號": "No",
    "球隊": "Team",
    "AVG": "BA",
    "BABIP": "BAbip",   # unify naming
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

# 3) MLB rename
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
mlb  = mlb.loc[mlb["BF"].replace([np.inf, -np.inf], np.nan)  >= BF_THRESHOLD].copy()
cpbl = cpbl.loc[cpbl["BF"].replace([np.inf, -np.inf], np.nan) >= BF_THRESHOLD].copy()
st.sidebar.write(f"MLB kept: {len(mlb)}/{mlb_before} | CPBL kept: {len(cpbl)}/{cpbl_before}")

# 5) Select columns (ADD BAbip as common)
common_cols = ["Player","Team","IP","BF","ERA","ERA_plus","FIP","WHIP",
               "K%","BB%","GB%","FB%","SO/BB","BAbip"]       # <-- BAbip added
mlb_unique  = ["WAR","HR9"]
cpbl_unique = ["Whiff%","Swing%","PutAway%"]
final_cols  = common_cols + mlb_unique + cpbl_unique

for col in final_cols:
    if col not in mlb.columns:  mlb[col]  = np.nan
    if col not in cpbl.columns: cpbl[col] = np.nan

mlb_clean  = mlb[final_cols].copy();  mlb_clean["League"]  = "MLB"
cpbl_clean = cpbl[final_cols].copy(); cpbl_clean["League"] = "CPBL"
combined = pd.concat([mlb_clean, cpbl_clean], ignore_index=True)

st.subheader("Combined Data (head)")
st.dataframe(combined.head())
st.subheader("Team counts")
st.dataframe(combined["Team"].value_counts(dropna=False).rename("count"))

# 6) KNN-impute WAR (ADD BAbip as feature)
ref_cols = [c for c in ["ERA","FIP","WHIP","K%","BB%","SO/BB","HR9","ERA_plus","BAbip"] if c in combined.columns]
if "WAR" in combined.columns and len(ref_cols) > 0:
    knn_data = combined[ref_cols + ["WAR"]].copy()
    knn_imp  = KNNImputer(n_neighbors=8, weights="distance")
    imputed  = knn_imp.fit_transform(knn_data)
    combined["WAR"] = imputed[:, -1]

# 7) Iterative imputation (team/league aware)
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

# Clip percentages & HR9
for c in [c for c in combined.columns if c.endswith("%")]:
    combined[c] = combined[c].clip(0, 100)
if "HR9" in combined.columns:
    combined["HR9"] = combined["HR9"].clip(lower=0)

na_left = combined.isna().sum()
st.write("Missing value summary (after imputation):",
         na_left[na_left > 0] if na_left.sum() > 0 else "All imputed.")

# 8) Correlation Heatmap (exclude encoded)
if show_heatmap:
    numeric_df = combined.select_dtypes(include=["float64","int64"]).replace([np.inf, -np.inf], np.nan)
    exclude_cols = ["Team_code","League_code"]
    if exclude_bf_in_heatmap:
        exclude_cols.append("BF")
    valid_corr_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 1 and c not in exclude_cols]
    corr = numeric_df[valid_corr_cols].corr(method="pearson")

    fig = plt.figure(figsize=(1 + 0.5*len(valid_corr_cols), 1 + 0.5*len(valid_corr_cols)))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", square=True,
                linewidths=0.5, cbar_kws={"shrink":0.8})
    plt.title("Pitching Variables Correlation Heatmap (Excluding Encoded Columns)")
    plt.tight_layout()
    st.pyplot(fig)

# 9) League KDE
if show_kde:
    st.subheader("Distribution Comparison by League (KDE)")
    plot_cols = [c for c in combined.select_dtypes(include=["float64","int64"]).columns
                 if c not in ["Team_code","League_code","BF"]]
    default_cols = ["ERA","ERA_plus","FIP","WHIP","K%","BB%","BAbip"]  # include BAbip
    selected = st.multiselect("Select variables", plot_cols,
                              default=[c for c in default_cols if c in plot_cols])
    if selected:
        cols = 3
        rows = int(np.ceil(len(selected)/cols))
        fig = plt.figure(figsize=(5*cols, 3.6*rows))
        for i, col in enumerate(selected):
            ax = plt.subplot(rows, cols, i+1)
            for lg in ["MLB","CPBL"]:
                sns.kdeplot(data=combined[combined["League"]==lg], x=col, fill=True, alpha=0.3, ax=ax, label=lg)
            ax.set_title(col); ax.legend()
        plt.suptitle("Distribution Comparison by League (After Imputation)", fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.97])
        st.pyplot(fig)

# 10) Team Violin
if show_violin:
    st.subheader(f"Team Violin Plot — {violin_metric}")
    plot_df = combined.dropna(subset=[violin_metric,"Team","League"]).copy()
    if facet_violin_by_league:
        for lg in ["MLB","CPBL"]:
            sub = plot_df[plot_df["League"]==lg]
            if sub.empty: continue
            fig = plt.figure(figsize=(max(10, 0.5*sub["Team"].nunique()), 6))
            sns.violinplot(data=sub, x="Team", y=violin_metric, inner="box",
                           linewidth=0.8, palette="tab20")
            plt.title(f"{violin_metric} Distribution by Team — {lg}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        fig = plt.figure(figsize=(max(10, 0.5*plot_df["Team"].nunique()), 6))
        sns.violinplot(data=plot_df, x="Team", y=violin_metric, inner="box",
                       linewidth=0.8, palette="tab20")
        plt.title(f"{violin_metric} Distribution by Team (All)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

# 11) ERA+ vs others (scatter grid)
if show_scatter:
    st.subheader("ERA+ vs Other Metrics (Scatter Grid)")
    candidates = [c for c in combined.select_dtypes(include=np.number).columns
                  if c not in ["ERA_plus","Team_code","League_code"]]
    # include BAbip automatically (already numeric)
    pick = st.multiselect("Pick variables (x-axis)", candidates,
                          default=candidates[:max_scatter])
    if pick:
        n = len(pick)
        cols = 5
        rows = int(np.ceil(n/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
        if rows == 1:
            axes = np.array([axes])
        palette = sns.color_palette("tab20", n_colors=combined["Team"].nunique())
        for i, xcol in enumerate(pick):
            ax = axes[i//cols, i%cols]
            sns.scatterplot(data=combined, x=xcol, y="ERA_plus",
                            hue="Team", s=50, alpha=0.7, ax=ax, palette=palette, legend=False)
            ax.set_title(xcol)
        for j in range(i+1, rows*cols):
            fig.delaxes(axes[j//cols, j%cols])
        plt.tight_layout()
        st.pyplot(fig)

# 12) Download processed CSV
st.subheader("Download processed CSV")
buf = io.BytesIO()
combined.to_csv(buf, index=False, encoding="utf-8-sig")
st.download_button("Download combined_pitching_processed.csv", buf.getvalue(),
                   file_name="combined_pitching_processed.csv", mime="text/csv")
