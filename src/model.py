"""Step 2: Model training, comparison and evaluation."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, joblib, json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, classification_report)
from utils import load_clean, FIG, ROOT

df = load_clean()
X, y = df.drop(columns="Churn"), df["Churn"]
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

rows, fitted = [], {}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
plt.figure(figsize=(6, 5))
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)]).fit(X_tr, y_tr)
    pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
                 "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
                 "ROC-AUC": roc_auc_score(y_te, proba), "CV ROC-AUC": cv_auc})
    fitted[name] = pipe
    fpr, tpr, _ = roc_curve(y_te, proba); plt.plot(fpr, tpr, label=f"{name} (AUC={rows[-1]['ROC-AUC']:.2f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(); plt.savefig(FIG / "06_roc_curves.png", dpi=150, bbox_inches="tight"); plt.close()

res = pd.DataFrame(rows).set_index("Model").round(3)
print("\n=== MODEL COMPARISON ===\n", res)
res.to_csv(ROOT / "reports" / "model_comparison.csv")

best_name = res["ROC-AUC"].idxmax(); best = fitted[best_name]
print(f"\nBest model: {best_name}\n")
print(classification_report(y_te, best.predict(X_te), target_names=["Stayed", "Churned"]))

cm = confusion_matrix(y_te, best.predict(X_te))
plt.figure(figsize=(4.5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
plt.title(f"Confusion Matrix - {best_name}"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.savefig(FIG / "07_confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

rf = fitted["Random Forest"]
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values().tail(15)
imp.index = [i.split("__")[1] for i in imp.index]
plt.figure(figsize=(7, 6)); imp.plot.barh(color="#E4572E"); plt.title("Top 15 Churn Drivers (Random Forest)")
plt.savefig(FIG / "08_feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()
print("Top drivers:", list(imp.index[::-1][:5]))

joblib.dump(best, ROOT / "models" / "best_model.joblib")
json.dump({"best_model": best_name, "columns": X.columns.tolist()}, open(ROOT / "models" / "meta.json", "w"))
print("Saved model -> models/best_model.joblib")
