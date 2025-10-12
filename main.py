import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load Data ===
cpbl = pd.read_excel('投手.xlsx')
mlb = pd.read_csv('MLB_Pitch.csv')

# === Data Processing Block ===
# 1) Create SO/BB in CPBL (avoid division by zero)
cpbl['SO/BB'] = (cpbl['K%'] / cpbl['BB%']).replace([np.inf, -np.inf], np.nan)

# 2) Rename CPBL columns to align with MLB
rename_map_cpbl = {
    "球員": "Player",
    "背號": "No",
    "球隊": "Team",  # keep team info for analysis
    "AVG": "BA",
    "BABIP": "BAbip",            # <─ 這裡已經把 CPBL 的 BABIP 統一成 BAbip
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
cpbl.rename(columns=rename_map_cpbl, inplace=True)

# 3) Rename MLB columns for consistency
mlb.rename(columns={"ERA+": "ERA_plus"}, inplace=True)

# Ensure MLB also has a 'Team' column
if 'Team' not in mlb.columns:
    print("MLB dataset missing 'Team' column — attempting to use fallback columns.")
    if 'Tm' in mlb.columns:
        mlb.rename(columns={'Tm': 'Team'}, inplace=True)
    else:
        mlb['Team'] = 'Unknown'

# 4) Drop pitchers with too few batters faced
BF_THRESHOLD = 70
cpbl['BF'] = pd.to_numeric(cpbl['BF'], errors='coerce')
mlb['BF'] = pd.to_numeric(mlb['BF'], errors='coerce')

cpbl_before, mlb_before = len(cpbl), len(mlb)
cpbl = cpbl[cpbl['BF'].replace([np.inf, -np.inf], np.nan) >= BF_THRESHOLD].copy()
mlb  = mlb[mlb['BF'].replace([np.inf, -np.inf], np.nan) >= BF_THRESHOLD].copy()
cpbl_after, mlb_after = len(cpbl), len(mlb)

print(f"Removed {cpbl_before - cpbl_after} CPBL pitchers with BF < {BF_THRESHOLD}. Remaining: {cpbl_after}")
print(f"Removed {mlb_before - mlb_after} MLB pitchers with BF < {BF_THRESHOLD}. Remaining: {mlb_after}")

# === Step 3. Define Columns to Merge ===
# 加入 BAbip 作為共通欄位
common_cols = ["Player", "Team", "IP", "BF", "ERA", "ERA_plus", "FIP", "WHIP",
               "K%", "BB%", "GB%", "FB%", "SO/BB", "BAbip"]
mlb_unique  = ["WAR", "HR9"]
cpbl_unique = ["Whiff%", "Swing%", "PutAway%"]
final_cols = common_cols + mlb_unique + cpbl_unique

# === Step 4. Ensure Structural Consistency ===
for col in final_cols:
    if col not in mlb.columns:
        mlb[col] = np.nan
    if col not in cpbl.columns:
        cpbl[col] = np.nan

# === Step 5. Add League Label + Select Columns ===
mlb_clean = mlb[final_cols].copy()
mlb_clean["League"] = "MLB"
cpbl_clean = cpbl[final_cols].copy()
cpbl_clean["League"] = "CPBL"

# === Step 6. Combine Datasets ===
combined = pd.concat([mlb_clean, cpbl_clean], ignore_index=True)
combined.info()

team_counts = combined['Team'].value_counts(dropna=False)
print("\nCombined dataset summary by Team:")
for team, count in team_counts.items():
    print(f"  {team}: {count} pitchers")

# === Step 7a. Re-impute WAR Using KNN ===
from sklearn.impute import KNNImputer

print("\nImputing WAR using KNNImputer for smoother distribution...")
# 把 BAbip 也納入參考特徵
ref_cols = ["ERA", "FIP", "WHIP", "K%", "BB%", "SO/BB", "HR9", "ERA_plus", "BAbip"]
ref_cols = [c for c in ref_cols if c in combined.columns]

cols_for_war = ref_cols + ["WAR"]
knn_data = combined[cols_for_war].copy()

knn_imp = KNNImputer(n_neighbors=8, weights="distance")
imputed = knn_imp.fit_transform(knn_data)
combined["WAR"] = imputed[:, -1]
print("WAR imputation completed successfully.")

# === Step 7b. Iterative Imputation (Cross-League + Team aware) ===
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Numeric encodings to inform imputer (not for analysis)
combined["Team_code"] = pd.factorize(combined["Team"])[0].astype(float)
combined["League_code"] = (combined["League"] == "MLB").astype(float)

numeric_cols = combined.select_dtypes(include=["float64", "int64"]).columns.tolist()
X = combined[numeric_cols].replace([np.inf, -np.inf], np.nan)

valid_cols = [c for c in X.columns if X[c].notna().sum() > 0]
dropped_cols = [c for c in X.columns if c not in valid_cols]
if dropped_cols:
    print(f"Skipped columns with all missing values: {dropped_cols}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[valid_cols])

imp = IterativeImputer(max_iter=30, random_state=42, tol=1e-3)
X_imp_scaled = imp.fit_transform(X_scaled)
X_imp = scaler.inverse_transform(X_imp_scaled)

X_filled = X.copy()
X_filled[valid_cols] = X_imp
combined[numeric_cols] = X_filled[numeric_cols]

# Clip outliers in percentages and ensure HR9 non-negative
percent_cols = [c for c in combined.columns if c.endswith("%")]
for c in percent_cols:
    combined[c] = combined[c].clip(lower=0, upper=100)
if "HR9" in combined.columns:
    combined["HR9"] = combined["HR9"].clip(lower=0)

print("\nMissing value summary after imputation:")
na_left = combined.isna().sum()
if na_left.sum() == 0:
    print("All missing values have been successfully imputed.")
else:
    print(na_left[na_left > 0])

# === Step 8. Correlation Heatmap (After Imputation, excluding encoded columns) ===
numeric_df = combined.select_dtypes(include=["float64", "int64"]).replace([np.inf, -np.inf], np.nan)

# Exclude helper encodings from the correlation view
exclude_cols = ["Team_code", "League_code", "BF"]  # also drop BF if you don't want volume to dominate
valid_corr_cols = [c for c in numeric_df.columns
                   if numeric_df[c].notna().sum() > 1 and c not in exclude_cols]

corr_matrix = numeric_df[valid_corr_cols].corr(method="pearson")

plt.figure(figsize=(1 + 0.5 * len(valid_corr_cols), 1 + 0.5 * len(valid_corr_cols)))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Pitching Variables Correlation Heatmap (Excluding Encoded Columns)")
plt.tight_layout()
plt.show()
plt.close()

# === Distribution Comparison (KDE) ===
print("\nPlotting variable distributions by League (after imputation)...")
plot_cols = [c for c in combined.select_dtypes(include=["float64", "int64"]).columns
             if c not in ["Team_code", "League_code", "BF"]]
cols = 3
rows = int(np.ceil(len(plot_cols) / cols))
plt.figure(figsize=(5 * cols, 3.6 * rows))

for i, col in enumerate(plot_cols):
    plt.subplot(rows, cols, i + 1)
    for lg in ["MLB", "CPBL"]:
        sns.kdeplot(
            data=combined[combined["League"] == lg],
            x=col,
            label=lg,
            fill=True,
            alpha=0.3
        )
    plt.title(col)
    plt.legend()

plt.suptitle("Distribution Comparison by League (After Imputation)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
plt.close()

# === Optional Boxplot for Overall Check ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=combined[plot_cols], orient="h", fliersize=2)
plt.title("Boxplot of Numeric Variables After Imputation")
plt.tight_layout()
plt.show()
plt.close()

# === Step 9. Violin Plot of ERA+ by Team ===
plt.figure(figsize=(24,10))
sns.violinplot(
    data=combined,
    x='Team',
    y='ERA_plus',
    palette='Set2',
    inner='box',
    width=0.9,
    hue='Team'
)
plt.xticks(rotation=90)
plt.xlabel('Team', fontsize=12)
plt.ylabel('ERA+', fontsize=12)
plt.title('ERA+/Team', fontsize=16)
plt.tight_layout()
plt.show()

#  === Step 10. Scatter Plots of ERA+ vs Other Metrics Colored by Team ===
col = [x for x in combined.select_dtypes(include=np.number).columns
       if x not in ['ERA_plus', 'Team_code', 'League_code']]

fig, axes = plt.subplots(4, 5, figsize=(40, 24))
palette = sns.color_palette("tab20", n_colors=combined['Team'].nunique())

for i, j in enumerate(col):
    sns.scatterplot(
        data=combined,
        x=j,
        y='ERA_plus',
        hue='Team',
        s=60,
        palette=palette,
        alpha=0.7,
        ax=axes[i//5, i%5]
    )
    axes[i//5, i%5].set_title(j, fontsize=16)
    axes[i//5, i%5].tick_params(labelsize=12)
    axes[i//5, i%5].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=2, fontsize=10)

plt.tight_layout()
plt.show()
