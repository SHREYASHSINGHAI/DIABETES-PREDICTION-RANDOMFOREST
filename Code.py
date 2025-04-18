import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # Importing SMOTE

# GETTING THE DATASET
def load_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path)
        print("Dataset successfully loaded!")
        return dataset
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

# DATA CLEANING AND MISSING VALUE HANDLING
def cleaning_dataset(dataset):
    columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    imputer = SimpleImputer(missing_values=0, strategy="median")
    dataset[columns_to_fix] = imputer.fit_transform(dataset[columns_to_fix])
    return dataset

# FEATURE ENGINEERING
def feature_engineering(dataset):
    dataset["AgeGroup"] = pd.cut(dataset["Age"], bins=[20, 30, 40, 50, 60, 100], labels=False)
    return dataset

# PREPARE FEATURES AND LABELS (without normalization)
def prepare_features_labels(dataset):
    X = dataset[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "AgeGroup"]]
    y = dataset["Outcome"]
    print("Checking for null values in features:\n", X.isnull().sum())
    print("Dataset output (0:not diabetic , 1:diabetic) :\n", y.value_counts())
    return X, y

# HYPERPARAMETER TUNING AND CROSS-VALIDATION
def cross_validate_model(X, y):
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

    xgb = XGBClassifier( eval_metric='logloss', random_state=42)
    params = {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'scale_pos_weight': [scale_pos_weight]
    }

    grid = GridSearchCV(xgb, params, cv=5, scoring='f1_weighted')
    grid.fit(X, y)

    best_model = grid.best_estimator_

    scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(best_model, X, y, scoring=scoring, cv=10)

    print("\nCross-validation Results:")
    print("Precision (weighted):", scores['test_precision_weighted'].mean())
    print("Recall (weighted):", scores['test_recall_weighted'].mean())
    print("F1 Score (weighted):", scores['test_f1_weighted'].mean())

    return best_model.fit(X, y)

# METRICS AFTER TRAINING ON FULL DATA
def evaluate_final_model(model, X, y):
    y_pred = model.predict(X)
    print("\nFinal Evaluation on Resampled Data:")
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

# VISUALIZATION
def visualize_data(model, dataset, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.barh(range(X.shape[1]), importances[indices], align="center")
    plt.yticks(range(X.shape[1]), np.array(dataset.columns[:-1].tolist() + ['AgeGroup'])[indices])
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    sns.boxplot(x="Outcome", y="Pregnancies", data=dataset, palette="pastel", hue="Outcome")
    plt.title("Pregnancy-Diabetes Relation")
    plt.show()

    sns.jointplot(data=dataset, x="Glucose", y="Insulin", kind="scatter", hue="Outcome", height=8)
    plt.suptitle("Glucose-Insulin Impact", y=1.02)
    plt.show()

    sns.scatterplot(x="BMI", y="SkinThickness", data=dataset, hue="Outcome", palette="colorblind")
    plt.title("Skin Thickness-BMI Relation")
    plt.show()

# MAIN
def main():
    current_dir = os.getcwd()
    input_file = os.path.join(current_dir, "DIABETES/DIABETES-PREDICTION-RANDOMFOREST/diabetes.csv")
    output_file = os.path.join(current_dir, "DIABETES/DIABETES-PREDICTION-RANDOMFOREST/Cleaned_diabetes.csv")

    dataset = load_dataset(input_file)
    if dataset is None:
        return

    dataset = cleaning_dataset(dataset)
    dataset = feature_engineering(dataset)
    dataset.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to {output_file}")

    X, y = prepare_features_labels(dataset)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Resampled dataset shape: {X_res.shape}, {y_res.shape}")

    model = cross_validate_model(X_res, y_res)
    evaluate_final_model(model, X_res, y_res)
    visualize_data(model, dataset, pd.DataFrame(X_res, columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                                              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "AgeGroup"]))

if __name__ == "__main__":
    main()
