# ============================================================
#   COVID-19 Early Case Trend Analysis & Recovery Insights
#   Company : HealthGuard Analytics Pvt. Ltd.
#   Project : Infectious Disease Case Analysis
# ============================================================

# ── STEP 1: Install & Import Libraries ───────────────────────
# Run in terminal before use:
#   pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Global plot style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor" : "#0d1117",
    "axes.facecolor"   : "#161b22",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#e6edf3",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "text.color"       : "#e6edf3",
    "grid.color"       : "#21262d",
    "figure.titlesize" : 14,
})
COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"]

print("=" * 60)
print("  COVID-19 Early Case Trend Analysis & Recovery Insights")
print("  HealthGuard Analytics Pvt. Ltd.")
print("=" * 60)


# ── STEP 2: Load Dataset ──────────────────────────────────────
# Download dataset from:
# https://drive.google.com/file/d/1TXoqikmE0S3LGem8Ig-GktaJJyZPGMgN/view?usp=sharing
# Save it as 'covid19_cases.csv' in the same folder as this script.

print("\n📂 Loading dataset...")
try:
    df = pd.read_csv("covid19_cases.csv")
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("⚠️  Dataset file not found. Generating synthetic demo data...")
    print("   Please download the real dataset from the Google Drive link in the project PDF.")
    np.random.seed(42)
    n = 500
    regions  = ["Seoul", "Gyeonggi", "Daegu", "Incheon", "Busan", "Gwangju"]
    reasons  = ["contact with patient", "overseas inflow", "gym facility",
                "religious gathering", "etc", "unknown"]
    outcomes = ["released", "isolated", "deceased"]

    confirmed = pd.date_range("2020-01-20", periods=n, freq="6H")
    released  = [c + pd.Timedelta(days=int(np.random.normal(20, 7)))
                 for c in confirmed]

    df = pd.DataFrame({
        "sex"              : np.random.choice(["male", "female"], n),
        "birth_year"       : np.random.randint(1940, 2005, n),
        "country"          : "Korea",
        "region"           : np.random.choice(regions, n),
        "infection_reason" : np.random.choice(reasons, n),
        "infection_order"  : np.random.randint(1, 5, n),
        "infected_by"      : np.random.randint(1000, 9999, n),
        "contact_number"   : np.random.randint(0, 50, n),
        "confirmed_date"   : [c.strftime("%Y-%m-%d") for c in confirmed],
        "released_date"    : [r.strftime("%Y-%m-%d") for r in released],
        "deceased_date"    : [np.nan] * n,
        "state"            : np.random.choice(outcomes, n, p=[0.70, 0.25, 0.05]),
    })
    print(f"✅ Synthetic dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")


# ── STEP 3: Exploratory Data Analysis (EDA) ──────────────────
print("\n── Exploratory Data Analysis ──")
print("\n📋 First 5 rows:")
print(df.head().to_string(index=False))

print("\n📊 Dataset Info:")
print(f"   Shape      : {df.shape}")
print(f"   Columns    : {list(df.columns)}")
print(f"   Data Types :\n{df.dtypes}")

print("\n🔍 Missing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing if not missing.empty else "   No missing values found.")

print("\n📈 Descriptive Statistics:")
print(df.describe(include='all').to_string())


# ── STEP 4: Data Cleaning & Feature Engineering ───────────────
print("\n⚙️  Cleaning & feature engineering...")

# Convert dates
df["confirmed_date"] = pd.to_datetime(df["confirmed_date"], errors="coerce")
df["released_date"]  = pd.to_datetime(df["released_date"],  errors="coerce")
df["deceased_date"]  = pd.to_datetime(df["deceased_date"],  errors="coerce")

# Compute age from birth_year
current_year = datetime.now().year
df["age"] = current_year - df["birth_year"]

# Compute recovery duration (days from confirmed to released)
df["recovery_days"] = (df["released_date"] - df["confirmed_date"]).dt.days

# Remove negative recovery days (data errors)
df = df[df["recovery_days"].isna() | (df["recovery_days"] >= 0)]

# Age groups
bins   = [0, 20, 40, 60, 80, 120]
labels = ["0–20", "21–40", "41–60", "61–80", "80+"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

print(f"✅ Feature engineering complete.")
print(f"   Age range        : {df['age'].min():.0f} – {df['age'].max():.0f} years")
print(f"   Recovery days    : {df['recovery_days'].min():.0f} – {df['recovery_days'].max():.0f} days")
print(f"   Avg recovery     : {df['recovery_days'].mean():.1f} days")
print(f"   Outcomes: {df['state'].value_counts().to_dict()}")


# ── STEP 5: VISUALIZATIONS ────────────────────────────────────

# ── 5A: Gender Distribution ───────────────────────────────────
print("\n📊 Generating: Gender Distribution chart...")
gender_counts = df["sex"].str.lower().value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Gender Distribution of COVID-19 Cases", fontsize=14,
             fontweight="bold", color="#e6edf3", y=1.02)

# Bar chart
axes[0].set_facecolor("#161b22")
bars = axes[0].bar(gender_counts.index, gender_counts.values,
                   color=["#58a6ff", "#f78166"], edgecolor="#30363d", linewidth=1.2)
for bar, val in zip(bars, gender_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(val), ha="center", fontweight="bold", color="#e6edf3")
axes[0].set_title("Case Count by Gender", color="#e6edf3")
axes[0].set_xlabel("Gender", color="#8b949e")
axes[0].set_ylabel("Number of Cases", color="#8b949e")

# Pie chart
axes[1].set_facecolor("#161b22")
axes[1].pie(gender_counts.values, labels=gender_counts.index,
            autopct="%1.1f%%", colors=["#58a6ff", "#f78166"],
            startangle=90, textprops={"color": "#e6edf3"})
axes[1].set_title("Gender Share (%)", color="#e6edf3")

plt.tight_layout()
plt.savefig("chart_01_gender_distribution.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✅ Saved: chart_01_gender_distribution.png")


# ── 5B: Age Distribution ──────────────────────────────────────
print("\n📊 Generating: Age Distribution chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Age Distribution of COVID-19 Patients", fontsize=14,
             fontweight="bold", color="#e6edf3")

# Histogram
axes[0].set_facecolor("#161b22")
axes[0].hist(df["age"].dropna(), bins=20, color="#58a6ff",
             edgecolor="#0d1117", linewidth=0.8)
axes[0].axvline(df["age"].mean(), color="#f78166", linestyle="--",
                linewidth=2, label=f"Mean age: {df['age'].mean():.1f}")
axes[0].set_title("Age Distribution (Histogram)", color="#e6edf3")
axes[0].set_xlabel("Age", color="#8b949e")
axes[0].set_ylabel("Count", color="#8b949e")
axes[0].legend(facecolor="#21262d", labelcolor="#e6edf3")

# Age group bar chart
axes[1].set_facecolor("#161b22")
age_grp = df["age_group"].value_counts().sort_index()
bars = axes[1].bar(age_grp.index.astype(str), age_grp.values,
                   color=COLORS[:len(age_grp)], edgecolor="#0d1117")
for bar, val in zip(bars, age_grp.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha="center", fontweight="bold", color="#e6edf3", fontsize=9)
axes[1].set_title("Cases by Age Group", color="#e6edf3")
axes[1].set_xlabel("Age Group", color="#8b949e")
axes[1].set_ylabel("Number of Cases", color="#8b949e")

plt.tight_layout()
plt.savefig("chart_02_age_distribution.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✅ Saved: chart_02_age_distribution.png")


# ── 5C: Case Outcome Distribution ────────────────────────────
print("\n📊 Generating: Case Outcomes chart...")
outcome_counts = df["state"].value_counts()
outcome_colors = {"released": "#3fb950", "isolated": "#ffa657", "deceased": "#f78166"}
colors = [outcome_colors.get(s, "#58a6ff") for s in outcome_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("COVID-19 Case Outcomes", fontsize=14,
             fontweight="bold", color="#e6edf3")

axes[0].set_facecolor("#161b22")
bars = axes[0].barh(outcome_counts.index, outcome_counts.values,
                    color=colors, edgecolor="#0d1117")
for bar, val in zip(bars, outcome_counts.values):
    axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontweight="bold", color="#e6edf3")
axes[0].set_title("Outcome Count", color="#e6edf3")
axes[0].set_xlabel("Cases", color="#8b949e")

axes[1].set_facecolor("#161b22")
axes[1].pie(outcome_counts.values, labels=outcome_counts.index,
            autopct="%1.1f%%", colors=colors, startangle=90,
            textprops={"color": "#e6edf3"})
axes[1].set_title("Outcome Share (%)", color="#e6edf3")

plt.tight_layout()
plt.savefig("chart_03_case_outcomes.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✅ Saved: chart_03_case_outcomes.png")


# ── 5D: Regional Case Concentration ──────────────────────────
print("\n📊 Generating: Regional Analysis chart...")
region_counts   = df["region"].value_counts().head(10)
region_released = df[df["state"] == "released"]["region"].value_counts()
region_deceased = df[df["state"] == "deceased"]["region"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Regional COVID-19 Case Concentration", fontsize=14,
             fontweight="bold", color="#e6edf3")

axes[0].set_facecolor("#161b22")
bars = axes[0].barh(region_counts.index[::-1], region_counts.values[::-1],
                    color="#58a6ff", edgecolor="#0d1117")
for bar, val in zip(bars, region_counts.values[::-1]):
    axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontsize=9, color="#e6edf3")
axes[0].set_title("Top Regions by Confirmed Cases", color="#e6edf3")
axes[0].set_xlabel("Cases", color="#8b949e")

# Stacked bar: confirmed vs released per region
top_regions = region_counts.index[:8]
x = np.arange(len(top_regions))
conf_vals = region_counts[top_regions].values
rel_vals  = [region_released.get(r, 0) for r in top_regions]
dec_vals  = [region_deceased.get(r, 0) for r in top_regions]

axes[1].set_facecolor("#161b22")
axes[1].bar(x, conf_vals, label="Confirmed", color="#58a6ff", alpha=0.8)
axes[1].bar(x, rel_vals,  label="Released",  color="#3fb950", alpha=0.8)
axes[1].bar(x, dec_vals,  label="Deceased",  color="#f78166", alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(top_regions, rotation=30, ha="right", fontsize=8)
axes[1].set_title("Confirmed vs Released vs Deceased by Region", color="#e6edf3")
axes[1].set_ylabel("Cases", color="#8b949e")
axes[1].legend(facecolor="#21262d", labelcolor="#e6edf3")

plt.tight_layout()
plt.savefig("chart_04_regional_analysis.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✅ Saved: chart_04_regional_analysis.png")


# ── 5E: Infection Sources ─────────────────────────────────────
print("\n📊 Generating: Infection Sources chart...")
infection_counts = df["infection_reason"].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Primary Infection Sources", fontsize=14,
             fontweight="bold", color="#e6edf3")

axes[0].set_facecolor("#161b22")
bars = axes[0].barh(infection_counts.index[::-1], infection_counts.values[::-1],
                    color=COLORS[:len(infection_counts)], edgecolor="#0d1117")
for bar, val in zip(bars, infection_counts.values[::-1]):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontsize=9, color="#e6edf3")
axes[0].set_title("Infection Reason Frequency", color="#e6edf3")
axes[0].set_xlabel("Cases", color="#8b949e")

axes[1].set_facecolor("#161b22")
axes[1].pie(infection_counts.values, labels=infection_counts.index,
            autopct="%1.1f%%", colors=COLORS[:len(infection_counts)],
            startangle=140, textprops={"color": "#e6edf3", "fontsize": 8})
axes[1].set_title("Share of Infection Sources", color="#e6edf3")

plt.tight_layout()
plt.savefig("chart_05_infection_sources.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✅ Saved: chart_05_infection_sources.png")


# ── 5F: Recovery Timeline ─────────────────────────────────────
print("\n📊 Generating: Recovery Timeline chart...")
recovery_df = df[df["recovery_days"].notna() & (df["state"] == "released")].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Recovery Timeline Analysis", fontsize=14,
             fontweight="bold", color="#e6edf3")

# Histogram of recovery days
axes[0].set_facecolor("#161b22")
axes[0].hist(recovery_df["recovery_days"], bins=25, color="#3fb950",
             edgecolor="#0d1117")
axes[0].axvline(recovery_df["recovery_days"].mean(), color="#f78166",
                linestyle="--", linewidth=2,
                label=f"Mean: {recovery_df['recovery_days'].mean():.1f} days")
axes[0].set_title("Distribution of Recovery Days", color="#e6edf3")
axes[0].set_xlabel("Days to Recovery", color="#8b949e")
axes[0].set_ylabel("Count", color="#8b949e")
axes[0].legend(facecolor="#21262d", labelcolor="#e6edf3")

# Recovery days by age group
axes[1].set_facecolor("#161b22")
age_recovery = recovery_df.groupby("age_group", observed=True)["recovery_days"].mean()
axes[1].bar(age_recovery.index.astype(str), age_recovery.values,
            color=COLORS[:len(age_recovery)], edgecolor="#0d1117")
axes[1].set_title("Avg Recovery Days by Age Group", color="#e6edf3")
axes[1].set_xlabel("Age Group", color="#8b949e")
axes[1].set_ylabel("Avg Days", color="#8b949e")

# Recovery days by gender
axes[2].set_facecolor("#161b22")
gender_recovery = recovery_df.groupby("sex")["recovery_days"].mean()
bars = axes[2].bar(gender_recovery.index, gender_recovery.values,
                   color=["#58a6ff", "#f78166"], edgecolor="#0d1117")
for bar, val in zip(bars, gender_recovery.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", fontweight="bold", color="#e6edf3")
axes[2].set_title("Avg Recovery Days by Gender", color="#e6edf3")
axes[2].set_xlabel("Gender", color="#8b949e")
axes[2].set_ylabel("Avg Days", color="#8b949e")

plt.tight_layout()
plt.savefig("chart_06_recovery_timeline.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✅ Saved: chart_06_recovery_timeline.png")


# ── 5G: Infection Order & Contact Exposure ───────────────────
print("\n📊 Generating: Infection Order & Contact Exposure chart...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Infection Order & Contact Exposure Analysis", fontsize=14,
             fontweight="bold", color="#e6edf3")

axes[0].set_facecolor("#161b22")
order_counts = df["infection_order"].value_counts().sort_index()
axes[0].bar(order_counts.index.astype(str), order_counts.values,
            color="#d2a8ff", edgecolor="#0d1117")
axes[0].set_title("Cases by Infection Order (Generation)", color="#e6edf3")
axes[0].set_xlabel("Infection Order", color="#8b949e")
axes[0].set_ylabel("Number of Cases", color="#8b949e")

axes[1].set_facecolor("#161b22")
axes[1].scatter(df["contact_number"], df["recovery_days"],
                alpha=0.5, color="#ffa657", edgecolors="#0d1117", linewidths=0.5)
axes[1].set_title("Contact Number vs Recovery Days", color="#e6edf3")
axes[1].set_xlabel("Contact Number", color="#8b949e")
axes[1].set_ylabel("Recovery Days", color="#8b949e")

plt.tight_layout()
plt.savefig("chart_07_infection_order_contact.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✅ Saved: chart_07_infection_order_contact.png")


# ── 5H: Correlation Heatmap ───────────────────────────────────
print("\n📊 Generating: Correlation Heatmap...")
corr_cols = ["age", "contact_number", "infection_order", "recovery_days"]
corr_df   = df[corr_cols].dropna()
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, linecolor="#30363d",
            annot_kws={"color": "#e6edf3"})
ax.set_title("Correlation Heatmap — Key Variables", color="#e6edf3",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("chart_08_correlation_heatmap.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✅ Saved: chart_08_correlation_heatmap.png")

print(f"\n📊 Correlation with Recovery Days:\n{corr_matrix['recovery_days'].drop('recovery_days')}")


# ── STEP 6: LINEAR REGRESSION ────────────────────────────────
print("\n" + "=" * 60)
print("  LINEAR REGRESSION — Predicting Recovery Time")
print("=" * 60)

# ── 6A: Feature Preparation ──────────────────────────────────
reg_df = df[["age", "contact_number", "infection_order",
             "sex", "region", "recovery_days"]].dropna().copy()

# Encode categorical columns
le_sex    = LabelEncoder()
le_region = LabelEncoder()
reg_df["sex_enc"]    = le_sex.fit_transform(reg_df["sex"])
reg_df["region_enc"] = le_region.fit_transform(reg_df["region"])

features = ["age", "contact_number", "infection_order", "sex_enc", "region_enc"]
X = reg_df[features]
y = reg_df["recovery_days"]

print(f"\nRegression dataset size: {len(reg_df)} records")
print(f"Features used : {features}")

# ── 6B: Train / Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train samples : {len(X_train)} | Test samples : {len(X_test)}")

# ── 6C: Fit Model ────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── 6D: Model Evaluation ─────────────────────────────────────
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n── Model Performance ──")
print(f"   R² Score : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"   MAE      : {mae:.2f} days")
print(f"   RMSE     : {rmse:.2f} days")

# ── 6E: Feature Coefficients ─────────────────────────────────
coef_df = pd.DataFrame({
    "Feature"     : features,
    "Coefficient" : model.coef_
}).sort_values("Coefficient", ascending=False)
print(f"\n── Feature Coefficients ──")
print(coef_df.to_string(index=False))
print(f"   Intercept: {model.intercept_:.4f}")


# ── 6F: Regression Visualizations ───────────────────────────
print("\n📊 Generating: Regression Analysis charts...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Linear Regression — Predicting Recovery Days", fontsize=14,
             fontweight="bold", color="#e6edf3")

# Actual vs Predicted
axes[0].set_facecolor("#161b22")
axes[0].scatter(y_test, y_pred, alpha=0.6, color="#58a6ff",
                edgecolors="#0d1117", linewidths=0.5)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val],
             color="#f78166", linestyle="--", linewidth=2, label="Perfect fit")
axes[0].set_title(f"Actual vs Predicted\nR² = {r2:.3f}", color="#e6edf3")
axes[0].set_xlabel("Actual Recovery Days", color="#8b949e")
axes[0].set_ylabel("Predicted Recovery Days", color="#8b949e")
axes[0].legend(facecolor="#21262d", labelcolor="#e6edf3")

# Residuals
residuals = y_test - y_pred
axes[1].set_facecolor("#161b22")
axes[1].scatter(y_pred, residuals, alpha=0.6, color="#3fb950",
                edgecolors="#0d1117", linewidths=0.5)
axes[1].axhline(0, color="#f78166", linestyle="--", linewidth=2)
axes[1].set_title("Residual Analysis", color="#e6edf3")
axes[1].set_xlabel("Predicted Values", color="#8b949e")
axes[1].set_ylabel("Residuals (Actual − Predicted)", color="#8b949e")

# Feature Importance
axes[2].set_facecolor("#161b22")
bar_colors = ["#3fb950" if c >= 0 else "#f78166" for c in coef_df["Coefficient"]]
axes[2].barh(coef_df["Feature"], coef_df["Coefficient"],
             color=bar_colors, edgecolor="#0d1117")
axes[2].axvline(0, color="#e6edf3", linewidth=0.8)
axes[2].set_title("Feature Coefficients\n(Impact on Recovery Days)", color="#e6edf3")
axes[2].set_xlabel("Coefficient Value", color="#8b949e")

plt.tight_layout()
plt.savefig("chart_09_regression_analysis.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✅ Saved: chart_09_regression_analysis.png")


# ── STEP 7: SUMMARY REPORT ───────────────────────────────────
print("\n" + "=" * 60)
print("  📋 FINAL ANALYSIS SUMMARY REPORT")
print("=" * 60)

total         = len(df)
released_pct  = (df["state"] == "released").sum() / total * 100
isolated_pct  = (df["state"] == "isolated").sum() / total * 100
deceased_pct  = (df["state"] == "deceased").sum() / total * 100
top_region    = df["region"].value_counts().index[0]
top_reason    = df["infection_reason"].value_counts().index[0]
dominant_age  = df["age_group"].value_counts().index[0]

print(f"""
  Company   : HealthGuard Analytics Pvt. Ltd.
  Project   : COVID-19 Early Case Trend Analysis

  ── Dataset ─────────────────────────────────────
  Total Records    : {total}
  Date Range       : {df['confirmed_date'].min().date()} → {df['confirmed_date'].max().date()}

  ── Demographics ─────────────────────────────────
  Most Affected Age Group : {dominant_age} years
  Average Patient Age     : {df['age'].mean():.1f} years
  Gender Split            : {df['sex'].value_counts().to_dict()}

  ── Infection Spread ──────────────────────────────
  Top Infection Source    : {top_reason}
  Most Impacted Region    : {top_region}
  Avg Contact Exposure    : {df['contact_number'].mean():.1f} contacts

  ── Outcomes ─────────────────────────────────────
  Released  : {released_pct:.1f}%
  Isolated  : {isolated_pct:.1f}%
  Deceased  : {deceased_pct:.1f}%

  ── Recovery ─────────────────────────────────────
  Average Recovery Time   : {df['recovery_days'].mean():.1f} days
  Fastest Recovery        : {df['recovery_days'].min():.0f} days
  Longest Recovery        : {df['recovery_days'].max():.0f} days

  ── Linear Regression ─────────────────────────────
  R² Score  : {r2:.4f}  ({r2*100:.1f}% variance explained)
  MAE       : {mae:.2f} days
  RMSE      : {rmse:.2f} days
  Top Factor: {coef_df.iloc[0]['Feature']} (coef: {coef_df.iloc[0]['Coefficient']:.3f})
""")

print("=" * 60)
print("🎉 Analysis complete! Charts saved:")
charts = [
    "chart_01_gender_distribution.png",
    "chart_02_age_distribution.png",
    "chart_03_case_outcomes.png",
    "chart_04_regional_analysis.png",
    "chart_05_infection_sources.png",
    "chart_06_recovery_timeline.png",
    "chart_07_infection_order_contact.png",
    "chart_08_correlation_heatmap.png",
    "chart_09_regression_analysis.png",
]
for c in charts:
    print(f"   📊 {c}")
print("=" * 60)
