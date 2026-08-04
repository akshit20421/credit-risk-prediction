# Credit Risk Prediction System

A machine learning-based credit risk classification system that predicts whether a consumer represents **Good or Bad credit risk** based on demographic and financial attributes. The project covers the complete workflow from exploratory data analysis and preprocessing to model comparison, hyperparameter tuning, and deployment through an interactive Streamlit application.

## Overview

Credit risk assessment is an important part of lending decisions. This project applies machine learning techniques to analyze applicant characteristics and classify consumers based on their credit risk.

The final system uses a **Random Forest Classifier**, which achieved the highest test accuracy among the evaluated models.

## Features Used

The model uses the following applicant information:

* Age
* Sex
* Job
* Housing
* Saving Accounts
* Checking Account
* Credit Amount
* Loan Duration

Categorical variables are encoded before being passed to the trained model. The Streamlit application uses the same saved encoders during inference to maintain consistency with model training.

## Project Workflow

```text
Data Collection
      ↓
Exploratory Data Analysis
      ↓
Missing Value Handling
      ↓
Categorical Encoding
      ↓
Train-Test Split
      ↓
Model Training
      ↓
5-Fold GridSearchCV
      ↓
Model Comparison
      ↓
Random Forest Selection
      ↓
Model Serialization
      ↓
Streamlit Application
```

## Machine Learning Models

Four tree-based classification algorithms were evaluated:

| Model         | Test Accuracy |
| ------------- | ------------: |
| Decision Tree |         70.5% |
| Random Forest |     **76.5%** |
| Extra Trees   |         74.5% |
| XGBoost       |         73.0% |

**Random Forest achieved the highest test accuracy of 76.5%** and was selected for the final prediction application.

## Model Training

The dataset was divided into training and testing sets using an **80:20 stratified split**.

Hyperparameter optimization was performed using **GridSearchCV with 5-fold cross-validation**.

The workflow compared:

* Decision Tree
* Random Forest
* Extra Trees
* XGBoost

The best-performing Random Forest model was serialized using Joblib for use in the Streamlit application.

## Streamlit Application

An interactive web application was developed using **Streamlit**.

Users can provide applicant information such as:

```text
Age
Sex
Job
Housing
Saving Account Status
Checking Account Status
Credit Amount
Loan Duration
```

The application preprocesses the inputs using the saved categorical encoders and passes them to the trained Random Forest model.

The final output classifies the applicant as:

```text
GOOD Credit Risk
```

or

```text
BAD Credit Risk
```

## Project Structure

```text
credit-risk-prediction/
│
├── app.py
├── analysis_model.ipynb
├── german_credit_data.csv
├── README.md
├── requirements.txt
│
└── models/
    ├── random_forest_credit_model.pkl
    ├── Sex_label_encoder.pkl
    ├── Housing_label_encoder.pkl
    ├── Saving accounts_label_encoder.pkl
    ├── Checking account_label_encoder.pkl
    └── target_encoder.pkl
```

> If the model files are stored inside the `models/` directory, make sure the corresponding paths in `app.py` point to the `models/` directory.

## Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**
* **Streamlit**
* **Joblib**

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd credit-risk-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Alternatively:

```bash
python -m streamlit run app.py
```

The application will normally be available locally at:

```text
http://localhost:8501
```

## Model Output

The trained classifier predicts one of two classes:

* **Good Risk** — applicant is predicted to represent lower credit risk.
* **Bad Risk** — applicant is predicted to represent higher credit risk.

The prediction is based on the combined pattern of applicant attributes rather than a single fixed credit-risk threshold.

## Future Improvements

Potential extensions to the project include:

* ROC-AUC, precision, recall and F1-score based model evaluation
* Confusion matrix and class-specific performance analysis
* Credit-risk probability scores instead of only binary predictions
* SHAP-based model explainability
* Improved categorical feature encoding
* Feature engineering and feature selection
* Probability calibration
* Streamlit Cloud deployment

## Disclaimer

This project is intended for **educational and portfolio purposes**. The model should not be used to make real-world lending or financial decisions without additional validation, fairness testing, explainability analysis, regulatory review, and appropriate risk controls.
