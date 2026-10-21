# Likely Viva Questions with Short Answers

1. **What is churn?** Customers discontinuing a service in a given period.
2. **Why is accuracy not enough?** Data is imbalanced (~27% churn); a model predicting "no churn" always gets ~73% accuracy. Use recall, precision, F1, ROC-AUC.
3. **Why did you convert TotalCharges?** It was stored as text with blank strings for tenure-0 customers; models need numeric input.
4. **What is one-hot encoding?** Converting each category into binary 0/1 columns so models can use it.
5. **Why scale numeric features?** Logistic regression is sensitive to feature scale; scaling helps convergence.
6. **What is stratified splitting?** Keeps the churn ratio the same in train and test sets.
7. **What is cross-validation?** Training/testing on k different folds to get a more reliable performance estimate.
8. **What does ROC-AUC mean?** Probability the model ranks a random churner above a random non-churner; 0.5 = random, 1 = perfect.
9. **Precision vs recall?** Precision = of predicted churners, how many truly churned. Recall = of actual churners, how many we caught.
10. **Why class_weight='balanced'?** Penalises mistakes on the minority class more so the model does not ignore churners.
11. **Why did Logistic Regression do well?** Relationships are largely linear and the data is tabular and small; it is also interpretable.
12. **How does Random Forest work?** Many decision trees trained on random subsets; predictions are averaged (bagging).
13. **How does Gradient Boosting differ?** Trees are built sequentially, each correcting the previous one's errors.
14. **What is overfitting and how did you check?** Model memorises training data; checked via test-set and CV scores.
15. **What is feature engineering?** Creating new informative variables (tenure_group, num_services).
16. **How would you deploy this?** Save model with joblib, serve via Streamlit/Flask/FastAPI, score customers monthly.
17. **What would you improve?** Hyperparameter tuning, SMOTE, SHAP, more behavioural data (call logs, complaints).
18. **Business impact?** Target high-risk customers with retention offers; revenue-at-risk shown in dashboard.
19. **Is your data real?** (Be honest: state whether you used the Kaggle dataset or the synthetic generator.)
