# WAEC_Likelihood_Predictor_Capstone_Project

Predicting whether a student is likely to **Pass** or **Fail** the WAEC (West African Examinations Council) exam using academic, behavioral, and socioeconomic data.

## Objective

To predict whether a student is likely to **Pass** or **Fail** the WAEC exam using academic performance, study habits, and socioeconomic indicators — enabling early identification of at-risk students for targeted intervention.

## Overview

This project builds and compares supervised classification models trained on student-level data, including study hours, attendance, prior exam scores, parental education, household income, school type, and access to resources such as internet and textbooks.

The pipeline covers:
- Data cleaning and missing-value checks
- Preprocessing (imputation, scaling, one-hot encoding)
- Model training (Logistic Regression, Random Forest)
- Evaluation (ROC-AUC, precision-recall, confusion matrix, classification report)
- Feature importance interpretation
- Model persistence with Joblib

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data handling | Pandas, NumPy |
| Modeling | scikit-learn (Logistic Regression, Random Forest Classifier) |
| Preprocessing | ColumnTransformer, Pipeline, SimpleImputer, StandardScaler, OneHotEncoder |
| Evaluation | ROC-AUC, Average Precision, Confusion Matrix, Classification Report |
| Visualization | Matplotlib, Seaborn |
| Model persistence | Joblib |
| Environment | Google Colab / Jupyter Notebook |

## Dataset

`waec_pass_likelihood_dataset.csv` — student-level records with the following feature groups:

- **Demographics:** `Age`, `Gender`, `State`, `School_Type`
- **Academic factors:** `Study_Hours_Per_Week`, `Attendance_Rate_Pct`, `Previous_Exam_Score_Pct`, `Mock_Exam_Score_Pct`, `Number_of_Subjects`
- **Support factors:** `Attends_Extra_Classes`, `Parental_Education_Level`, `Household_Income_Level`, `Distance_to_School_km`, `Has_Internet_Access`, `Has_Textbook_Access`
- **Target:** `Predicted_Outcome` (Pass/Fail), with `WAEC_Pass_Likelihood_Pct` as the underlying probability score

## Repository Structure

```
├── WAEC_Likelihood.ipynb          # Main notebook (Colab-generated)
├── waec_likelihood.py             # Script export of the notebook
├── waec_pass_likelihood_dataset.csv  # Training dataset
├── final_model.pkl                # Saved trained model (generated on run)
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd waec-pass-likelihood
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

Run the script directly:

```bash
python waec_likelihood.py
```

Or open the notebook in Jupyter/Colab:

```bash
jupyter notebook WAEC_Likelihood.ipynb
```

The script will:
1. Load and inspect the dataset
2. Split data into train/test sets (80/20)
3. Preprocess numeric and categorical features
4. Train Logistic Regression and Random Forest classifiers
5. Print evaluation metrics and display ROC/Precision-Recall curves
6. Save the final model to `final_model.pkl`
7. Plot feature importance from the Logistic Regression coefficients

## Results

| Model | ROC AUC | Accuracy | Avg Precision |
|---|---|---|---|
| Logistic Regression | 0.80 | 0.72 | 0.66 |
| Random Forest | 0.76 | 0.72 | 0.61 |

Logistic Regression slightly outperformed Random Forest on ROC-AUC and was selected as the final model. Both models reached ~72% accuracy, with stronger precision on predicting "Fail" outcomes than "Pass" — indicating room to improve recall on at-risk-but-passing students in future iterations.

## Future Improvements

- Address class imbalance (Fail: 76 vs Pass: 44 in test set) with techniques like SMOTE or class weighting
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for both models
- Try gradient boosting models (XGBoost, LightGBM) for comparison
- Cross-validation instead of a single train/test split for more robust metrics
- Deploy the final model via a simple API or web app for real-time predictions

## License

Specify your license here (e.g., MIT).
