# 🩺 Diabetes Prediction using Random Forest

This project aims to build a machine learning model that predicts the likelihood of diabetes in female patients using health-related metrics. The model uses the
**Random Forest Classifier**, which provides robustness, high accuracy, and interpretability through feature importance analysis.

## 📌 Problem Statement

The goal is to predict whether a female patient is likely to have diabetes based on medical diagnostic measurements. Early detection is crucial for prevention and proper treatment, especially in high-risk communities.

## 📁 Dataset

- **Name:** PIMA Indian Diabetes Dataset
- **Source:** Kaggle
- **Attributes:**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
  - Outcome (Target: 0 = No diabetes, 1 = Diabetes)

## 🧠 Features of Random Forest (Why it’s used here)

- Handles missing values and outliers effectively
- Works without feature scaling
- Captures non-linear patterns
- Reduces overfitting by combining multiple decision trees
- Provides feature importance to understand critical risk factors

## 🛠️ Tools & Libraries

- Python 3.x
- NumPy
- Pandas
- Matplotlib / Seaborn (for visualization)
- Scikit-learn (model training & evaluation)

## 🧪 Workflow

1. **Data Loading & Exploration**
2. **Data Cleaning & Missing Value Handling**
3. **Train-Test Splitting**
4. **Model Training using RandomForestClassifier**
5. **Model Evaluation (Accuracy, Confusion Matrix, F1-Score)**
6. **Feature Importance Visualization**

## 📈 Feature Importance

The Random Forest model reveals that **Glucose**, **BMI**, and **Diabetes Pedigree Function** are among the most influential features for predicting diabetes.
