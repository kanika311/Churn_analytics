"""Step 1: Cleaning + Exploratory Data Analysis. Saves charts to reports/figures/"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from utils import load_clean, FIG, DATA

sns.set_theme(style="whitegrid")
raw = pd.read_csv(DATA)
df = load_clean()
print("Raw shape:", raw.shape, "| Clean shape:", df.shape)
print("Missing values after cleaning:", int(df.isna().sum().sum()))
print("Churn rate: %.1f%%" % (df.Churn.mean() * 100))
print(df.describe().round(2))

fig, ax = plt.subplots(figsize=(5, 4))
df.Churn.map({0: "Stayed", 1: "Churned"}).value_counts().plot.pie(autopct="%1.1f%%", colors=["#4C9F70", "#E4572E"], ax=ax)
ax.set_ylabel(""); ax.set_title("Customer Churn Split"); fig.savefig(FIG / "01_churn_split.png", dpi=150, bbox_inches="tight"); plt.close()

cats = ["Contract", "InternetService", "PaymentMethod", "tenure_group", "SeniorCitizen", "TechSupport"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, c in zip(axes.ravel(), cats):
    (df.groupby(c).Churn.mean() * 100).sort_values().plot.barh(ax=ax, color="#2E86AB")
    ax.set_title(f"Churn % by {c}"); ax.set_xlabel("Churn rate (%)"); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(FIG / "02_churn_by_category.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, c in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.kdeplot(data=df, x=c, hue=df.Churn.map({0: "Stayed", 1: "Churned"}), fill=True, ax=ax, common_norm=False)
    ax.set_title(f"{c} vs Churn")
fig.tight_layout(); fig.savefig(FIG / "03_numeric_dist.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap"); fig.savefig(FIG / "04_correlation.png", dpi=150, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x=df.Churn.map({0: "Stayed", 1: "Churned"}), y=df["MonthlyCharges"], ax=ax)
ax.set_title("Monthly Charges by Churn"); ax.set_xlabel(""); fig.savefig(FIG / "05_boxplot.png", dpi=150, bbox_inches="tight"); plt.close()
print("EDA charts saved to reports/figures/")
