# 🦠 COVID-19 Early Case Trend Analysis & Recovery Insights
### Analytics Project | HealthGuard Analytics Pvt. Ltd.

---

## 📌 Project Overview

This is a full data analytics project built for **HealthGuard Analytics Pvt. Ltd.**,
a healthcare data analytics firm providing insights to government health departments
and hospitals.

The project analyzes early-stage COVID-19 patient-level data to answer 5 key
public health questions using EDA, descriptive statistics, data visualization,
and Linear Regression modeling.

---

## 📁 Project Structure

```
covid19-analysis/
│
├── covid19_analysis.py          ← Main Python script (full pipeline)
├── covid19_cases.csv            ← Dataset (download from Google Drive link)
│
├── chart_01_gender_distribution.png
├── chart_02_age_distribution.png
├── chart_03_case_outcomes.png
├── chart_04_regional_analysis.png
├── chart_05_infection_sources.png
├── chart_06_recovery_timeline.png
├── chart_07_infection_order_contact.png
├── chart_08_correlation_heatmap.png
├── chart_09_regression_analysis.png
│
└── README.md                    ← This file
```

---

## 📥 Dataset

**Download the dataset from:**
> https://drive.google.com/file/d/1TXoqikmE0S3LGem8Ig-GktaJJyZPGMgN/view?usp=sharing

Save the file as **`covid19_cases.csv`** in the same folder as `covid19_analysis.py`.

### Dataset Attributes

| Column | Description |
|--------|-------------|
| `sex` | Patient gender (male / female) |
| `birth_year` | Year of birth (used to compute age) |
| `country` | Country of the patient |
| `region` | Region / province of the patient |
| `infection_reason` | Primary source of infection |
| `infection_order` | Generation of infection (1st, 2nd...) |
| `infected_by` | Case ID of the source patient |
| `contact_number` | Number of contacts exposed |
| `confirmed_date` | Date the case was confirmed |
| `released_date` | Date the patient was released |
| `deceased_date` | Date of death (if applicable) |
| `state` | Outcome — released / isolated / deceased |

> **Note:** If the CSV file is not found, the script auto-generates a synthetic
> demo dataset so you can run and test the full pipeline immediately.

---

## ⚙️ Requirements & Installation

### Python Version
Python 3.8 or higher recommended.

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, and manipulation |
| `numpy` | Numerical operations and feature engineering |
| `matplotlib` | Chart generation and visualization |
| `seaborn` | Statistical visualizations (heatmaps, distributions) |
| `scikit-learn` | Linear Regression model and evaluation metrics |

---

## ▶️ How to Run

```bash
# Step 1 — Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Step 2 — Download dataset and save as covid19_cases.csv

# Step 3 — Run the full analysis
python covid19_analysis.py
```

All 9 charts will be saved as PNG files in the same directory.

---

## 🔍 Analysis Pipeline

```
  Load Dataset (CSV / Synthetic fallback)
           │
           ▼
  ┌──────────────────────────┐
  │  Step 3: EDA             │
  │  • Shape, dtypes         │
  │  • Missing values        │
  │  • Descriptive stats     │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Step 4: Data Cleaning   │
  │  • Date parsing          │
  │  • Age computation       │
  │  • Recovery days calc    │
  │  • Age group binning     │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Step 5: Visualizations  │
  │  • Gender distribution   │
  │  • Age distribution      │
  │  • Case outcomes         │
  │  • Regional analysis     │
  │  • Infection sources     │
  │  • Recovery timelines    │
  │  • Infection order       │
  │  • Correlation heatmap   │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Step 6: Linear          │
  │  Regression              │
  │  • Feature engineering   │
  │  • Train/Test split      │
  │  • Model training        │
  │  • R², MAE, RMSE         │
  │  • Residual analysis     │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Step 7: Summary Report  │
  │  • All KPIs printed      │
  │  • Charts listed         │
  └──────────────────────────┘
```

---

## 📊 Charts Generated (9 Total)

| Chart | What It Shows |
|-------|---------------|
| `chart_01_gender_distribution` | Bar + Pie chart of male vs female cases |
| `chart_02_age_distribution` | Age histogram + cases by age group |
| `chart_03_case_outcomes` | Released / Isolated / Deceased breakdown |
| `chart_04_regional_analysis` | Top regions + stacked outcome comparison |
| `chart_05_infection_sources` | Bar + Pie of primary infection reasons |
| `chart_06_recovery_timeline` | Recovery days distribution + by age/gender |
| `chart_07_infection_order_contact` | Infection generation + contact exposure scatter |
| `chart_08_correlation_heatmap` | Correlation between age, contacts, order, recovery |
| `chart_09_regression_analysis` | Actual vs Predicted + Residuals + Feature Coefficients |

---

## 🔑 Key Concepts Explained

| Concept | Explanation |
|---------|-------------|
| **EDA** | Exploratory Data Analysis — understanding data structure before modeling |
| **Descriptive Statistics** | Mean, median, std, min, max of variables |
| **Feature Engineering** | Creating new columns (age, recovery_days, age_group) |
| **Label Encoding** | Converting text categories (sex, region) to numbers for ML |
| **Linear Regression** | Predicts a continuous value (recovery days) from input features |
| **R² Score** | Proportion of variance explained by the model (0 to 1) |
| **MAE** | Mean Absolute Error — average error in days |
| **RMSE** | Root Mean Squared Error — penalizes large errors more |
| **Residuals** | Difference between actual and predicted values |
| **Correlation** | Statistical relationship strength between two variables (−1 to +1) |

---

## 📋 5 Business Questions Answered

| Question | Method Used |
|----------|-------------|
| 1. Who is getting infected? | Age & gender distribution charts |
| 2. How are infections spreading? | Infection reason & order analysis |
| 3. What are the recovery trends? | Recovery days distribution & timeline |
| 4. Which regions are most impacted? | Regional case concentration charts |
| 5. What factors influence recovery? | Correlation analysis + Linear Regression |

---

## 🚀 Optional Extensions (Already Included)

- ✅ Feature engineering for recovery duration (`recovery_days`)
- ✅ Linear Regression model to predict days-to-recovery
- ✅ Model evaluation using R² score, MAE, RMSE
- ✅ Residual analysis chart

---

## 🐛 Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError` | CSV not found | Download dataset and rename to `covid19_cases.csv` |
| `ModuleNotFoundError` | Library missing | `pip install <library>` |
| Empty recovery days | Dates missing in CSV | Script handles NaN automatically |
| Low R² score | Linear relationships are weak | This is expected — real health data is complex |
| Charts not displaying | No GUI environment | Charts are still saved as PNG files |

---

## 📚 Technologies Used

**Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn**

Statistical Methods: **Descriptive Statistics · Correlation Analysis · Linear Regression**

---

*Built for HealthGuard Analytics Pvt. Ltd. | Minor Project — AI & Data Analytics*
