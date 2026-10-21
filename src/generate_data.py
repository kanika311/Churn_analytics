"""Generates a realistic Telco-style customer churn dataset (7,043 rows).
If you have the real Kaggle 'Telco Customer Churn' CSV, save it as
data/telco_churn.csv and skip this script - the rest of the pipeline works on both."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 7043
ROOT = Path(__file__).resolve().parents[1]

gender = rng.choice(["Male", "Female"], n)
senior = rng.choice([0, 1], n, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
tenure = np.clip(rng.gamma(2.0, 16, n).astype(int), 0, 72)
phone = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
multi = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n))
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

def addon(p=0.4):
    return np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n, p=[p, 1 - p]))

security, backup, protect = addon(0.29), addon(0.34), addon(0.34)
support, tv, movies = addon(0.29), addon(0.38), addon(0.39)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                     n, p=[0.34, 0.23, 0.22, 0.21])

base = np.where(phone == "Yes", 20, 0) + np.select([internet == "DSL", internet == "Fiber optic"], [25, 55], 0)
extras = sum((x == "Yes") * 5 for x in [security, backup, protect, support, tv, movies])
monthly = np.round(base + extras + rng.normal(0, 3, n) + 5, 2).clip(18.25, 118.75)
total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)

z = (-1.1
     + 1.5 * (contract == "Month-to-month") - 1.0 * (contract == "Two year")
     + 0.8 * (internet == "Fiber optic") - 0.6 * (internet == "No")
     + 0.5 * (payment == "Electronic check")
     - 0.035 * tenure + 0.012 * (monthly - 65)
     - 0.5 * (security == "Yes") - 0.5 * (support == "Yes")
     + 0.3 * senior + 0.2 * (paperless == "Yes") - 0.2 * (partner == "Yes"))
churn = rng.random(n) < 1 / (1 + np.exp(-z))

ids = [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n)]
df = pd.DataFrame({
    "customerID": ids,
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
    "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protect,
    "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": total.astype(str), "Churn": np.where(churn, "Yes", "No"),
})
# real-world messiness: blank TotalCharges when tenure is 0 (same quirk as the Kaggle file)
df.loc[df.tenure == 0, "TotalCharges"] = " "
df.to_csv(ROOT / "data" / "telco_churn.csv", index=False)
print("Saved", df.shape, "| churn rate:", round((df.Churn == "Yes").mean() * 100, 1), "%")
