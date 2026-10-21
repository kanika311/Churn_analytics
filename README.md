# Customer Churn Prediction - Data Analytics Project

Analyse why telecom customers leave, predict who is likely to churn, and give business recommendations.

## Project structure
```
data/            telco_churn.csv            (dataset)
src/             generate_data.py, utils.py, eda.py, model.py
app/             streamlit_app.py           (interactive dashboard)
sql/             queries.sql                (SQL analysis)
notebooks/       churn_analysis.ipynb       (step-by-step notebook)
reports/         project_report.md, viva_questions.md, model_comparison.csv, figures/
models/          best_model.joblib
```

## How to run
```bash
pip install -r requirements.txt
cd src
python generate_data.py     # skip if you use the real Kaggle CSV
python eda.py               # cleaning + charts -> reports/figures/
python model.py             # trains 3 models, saves best one
cd ..
streamlit run app/streamlit_app.py
```

## IMPORTANT: dataset note
`generate_data.py` creates a **synthetic dataset that mimics the Kaggle "Telco Customer Churn"
dataset** (same columns, same size). For your submission, download the real one from Kaggle
(search "Telco Customer Churn", IBM sample data), save as `data/telco_churn.csv`, and re-run
`eda.py` and `model.py`. Then update the numbers in `reports/project_report.md` with YOUR results.
Do not present synthetic results as real-world findings.

## Pipeline
1. Cleaning: fix TotalCharges text -> number, remove duplicates, drop ID
2. Feature engineering: tenure_group, num_services, avg_monthly_spend
3. EDA: churn split, churn by category, distributions, correlation, boxplot
4. Modeling: Logistic Regression, Random Forest, Gradient Boosting (stratified split, 5-fold CV)
5. Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
6. Dashboard + SQL + business recommendations

## Tools
Python, Pandas, Scikit-learn, Matplotlib/Seaborn, Streamlit, SQL
