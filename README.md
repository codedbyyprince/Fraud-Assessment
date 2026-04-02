# Fraud-Assessment

A machine learning web application for detecting Card Not Present (CNP) fraud in transaction data. Upload a CSV of transactions, choose a model, and get back a scored file with fraud predictions and probability scores for each transaction.

Built by **Prince Nagda**.

Live demo: [fraud-detector-ml on Hugging Face Spaces](https://mlwithprince-fraud-detector-ml.hf.space/)

Test data: [drive-file](https://drive.google.com/file/d/1aAUkeEqb9iMLu2GpS8XS7oTol6thr7sl/view?usp=drive_link)
---

## What This Project Does

CNP fraud happens when a card is used without physical presence, typically in online transactions. The challenge is that most fraud detection datasets used in research contain features that are not available in real-world production systems. This project was built with that constraint in mind.

Every feature used for training was chosen based on whether it could realistically be obtained at transaction time. No synthetic or post-transaction data was used. The result is a model that is actually deployable, not just accurate on paper.

The application takes a raw transaction CSV, runs feature engineering internally, scores each transaction using one of four model options, and returns a downloadable CSV with predictions alongside a results viewer and a statistics dashboard.

---

## Models

All models are v4, trained on approximately 588,000 real-world CNP transactions derived from the IEEE-CIS Fraud Detection dataset, filtered and restructured to keep only features obtainable at transaction time.

**Core RF** is a Random Forest classifier. It tends toward higher precision, meaning when it flags something as fraud it is more likely to be right, but it will miss more fraud overall.

**Core XGB** is an XGBoost classifier. It offers a reasonable balance between catching fraud and avoiding false alarms, and generally has the best probability separation between fraud and non-fraud.

**Core LGB** is a LightGBM classifier. It leans toward higher recall, meaning it catches more fraud but will flag more legitimate transactions as suspicious.

**Apex 1.0** is the ensemble confidence model. It runs all three models on each transaction, computes an average fraud probability, and penalizes cases where the three models disagree strongly. The formula is: confidence score = average probability x (1 minus standard deviation of the three scores). This means a transaction that all three models agree on gets a high confidence score, while one where they disagree gets pulled down. The user sets the threshold at which the confidence score triggers a fraud prediction.

---

## Feature Engineering

The application handles all feature engineering internally. You do not need to pre-process your CSV. The pipeline normalizes browser and OS strings, derives time-based features from the transaction timestamp, computes amount bins, and builds environment risk scores from device, OS, and browser combinations.

The models were trained with this same pipeline, so the transformation applied at prediction time is consistent with training.

---

## Required CSV Columns

Your input CSV must contain the following columns. Column names are case-sensitive and must match exactly.

| Column | Type | Description |
|---|---|---|
| transaction_id | string or int | Unique identifier for each transaction |
| transaction_time | int | Unix timestamp of the transaction |
| transaction_amount | float | Amount of the transaction in any currency |
| card_network | string | Card network, e.g. Visa, Mastercard, Discover |
| card_type | string | Card type, e.g. credit, debit |
| purchaser_email_domain | string | Email domain of the buyer, e.g. gmail.com |
| device_type | string | Device used, e.g. mobile, desktop |
| is_identity_seen_before | string | Whether this identity has been seen, e.g. Found, New |
| user_os | string | Operating system of the user, e.g. Windows, iOS, Android |
| user_browser | string | Browser used, e.g. Chrome, Safari, Firefox |

Missing values in any column are handled gracefully. Unknown browser or OS strings are mapped to "Other" automatically.

---

## How to Use

**Step 1.** Go to the Home page. Select a model from the dropdown.

**Step 2.** If you selected a Core model (RF, XGB, or LGB), use the precision and recall slider to choose your operating threshold. Moving toward precision means fewer false alarms but more missed fraud. Moving toward recall means catching more fraud but more false alarms. The current precision and recall at your chosen threshold are shown in real time before you upload.

**Step 3.** If you selected Apex 1.0, set the confidence threshold as a percentage. Transactions whose confidence score meets or exceeds this value will be flagged as fraud. A threshold of 50 is a reasonable starting point.

**Step 4.** Upload your CSV and click Predict. Processing typically takes 30 to 50 seconds depending on file size. Do not close the tab while processing.

**Step 5.** After prediction completes, you are taken to the Results page. Here you can browse all transactions in a paginated table, see fraud probability and prediction for each row, and download the full scored CSV.

**Step 6.** Visit the Statistics page for a summary of the batch, including total transactions, total fraud flagged, fraud percentage, probability distribution across five risk bands, the five most suspicious transactions, and a threshold sensitivity chart showing how the flagged count changes across different thresholds.

---

## Output Columns Added

The output CSV contains all your original columns plus the following.

**fraud_probability** is a number from 0 to 100 representing the model's estimated probability that the transaction is fraudulent.

**fraud_prediction** is 0 or 1. A value of 1 means the transaction was flagged as fraud at the chosen threshold.

For Core models, three additional columns are included: threshold_used, precision_at_t, and recall_at_t, which record the threshold and its corresponding precision and recall at prediction time.

---

## Waiting Time

Prediction takes approximately 30 to 50 seconds for a typical CSV. Larger files will take longer. The page shows a live progress indicator during processing. Temporary files are stored server-side and automatically cleaned up after 12 hours.

---

## Limitations

The models were trained on a specific dataset and may not generalize perfectly to all transaction types or geographies. Recall across all models is intentionally bounded by the constraint of using only real-world obtainable features. Adding more proprietary signals such as device fingerprints, historical user behaviour, or IP data would improve performance significantly. This project demonstrates what is achievable with minimal but realistic feature sets.

Precision across models is low by design. In fraud detection, missing a fraud is generally more costly than a false alarm, so the models are tuned to prioritize recall over precision. Apex 1.0 provides a middle ground by requiring agreement across all three models before assigning a high confidence score.

---

## Project Structure

```
app.py          Flask application, routes, ML helpers, stats API
functions.py    Feature engineering pipeline and CSV validation
template/       HTML templates for home, results, and stats pages
requirements.txt  Python dependencies
Dockerfile      Container setup for Hugging Face Spaces
```

Models and PR curve data are loaded from a private Hugging Face model repository at runtime.

---

## Tech Stack

Python, Flask, scikit-learn, XGBoost, LightGBM, pandas, NumPy, Hugging Face Spaces, Docker.

---

## Author

Prince Nagda
