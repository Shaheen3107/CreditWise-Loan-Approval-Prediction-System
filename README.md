# CreditWise Loan Approval Prediction System

## Project Overview
The CreditWise Loan Approval Prediction System is a Machine Learning project that predicts whether a loan application should be approved or rejected based on customer financial and demographic information.

The goal of this project is to automate the loan approval process and assist financial institutions in making faster and more accurate lending decisions.

---

## Problem Statement
Banks receive thousands of loan applications, and manually evaluating them is time-consuming and prone to human error. This project uses machine learning algorithms to analyze applicant data and predict loan approval status.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Machine Learning Algorithms Implemented
The following algorithms were trained and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Naive Bayes

The models were compared using classification metrics to determine the best performing algorithm.

---

## Model Performance

### Logistic Regression (Best Model)
Accuracy: **87.5%**  
Precision: **0.79**  
Recall: **0.80**  
F1 Score: **0.79**

### KNN
Accuracy: **75.5%**

### Naive Bayes
Accuracy: **86.5%**

Logistic Regression performed the best and was selected as the final model.

---

## Project Workflow
1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Encoding
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Model Comparison
9. Saving the Best Model

---

## Project Structure

```
creditwise-loan-prediction/
│
├── Creditwise_Loan_system.ipynb
├── dataset.csv
├── model.pkl
├── requirements.txt
└── README.md
```

---

## How to Run the Project

1. Clone the repository

```
git clone https://github.com/Shaheen3107/CreditWise-Loan-Approval-Prediction-System
.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Open the notebook

```
jupyter notebook Creditwise_Loan_system.ipynb
```

---

## Future Improvements
- Hyperparameter tuning
- Model deployment using Flask or Streamlit
- Adding more advanced algorithms such as Random Forest and XGBoost

---

## Author

Shaheen Saiyad

Machine Learning Enthusiast