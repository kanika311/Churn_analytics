-- Load telco_churn.csv into a table named customers (SQLite / MySQL / PostgreSQL), then run:

-- 1. Overall churn rate
SELECT ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct FROM customers;

-- 2. Churn by contract type
SELECT Contract, COUNT(*) AS customers,
       ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_pct
FROM customers GROUP BY Contract ORDER BY churn_pct DESC;

-- 3. Monthly revenue lost to churn
SELECT ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_lost FROM customers WHERE Churn='Yes';

-- 4. Churn by tenure bucket
SELECT CASE WHEN tenure<=12 THEN '0-12' WHEN tenure<=24 THEN '13-24' WHEN tenure<=48 THEN '25-48' ELSE '49-72' END AS tenure_bucket,
       COUNT(*) AS customers, ROUND(100.0*AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END),2) AS churn_pct
FROM customers GROUP BY tenure_bucket ORDER BY tenure_bucket;

-- 5. High-risk segment: month-to-month + fiber + electronic check
SELECT COUNT(*) AS customers, ROUND(100.0*AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END),2) AS churn_pct
FROM customers WHERE Contract='Month-to-month' AND InternetService='Fiber optic' AND PaymentMethod='Electronic check';

-- 6. Does tech support reduce churn?
SELECT TechSupport, ROUND(100.0*AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END),2) AS churn_pct
FROM customers WHERE InternetService<>'No' GROUP BY TechSupport;
