#### TOO MUCHH BIASED DATASET ###########
import pandas as pd
import os
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score


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
    columns_to_fix = ["SkinThickness", "BMI", "DiabetesPedigreeFunction", "Age"]
    imputer = SimpleImputer(missing_values=0, strategy="median")
    dataset[columns_to_fix] = imputer.fit_transform(dataset[columns_to_fix])
    dataset.to_csv(r"DIABETES/DIABETES-PREDICTION-RANDOMFOREST/Cleaned_diabetes.csv", index=False)
    return dataset

# PREPARE FEATURES AND LABELS
def prepare_features_labels(dataset):
    X = dataset[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
    y = dataset["Outcome"]
    print("Checking for null values in features:\n", X.isnull().sum())
    print("Dataset output (0:not diabetic , 1:diabetic) : ",y.value_counts())
    return X, y

# CROSS-VALIDATION EVALUATION
def cross_validate_model(X, y):
    model = RandomForestClassifier(n_estimators=160, max_samples=0.74, max_depth=None, max_features=3,class_weight="balanced")
    scores = cross_val_score(model, X, y, cv=10, scoring='f1_weighted')#f1_weighted takes both precision and recall into account 
                                                                       #and averages them accounting for class imbalance.
                                                                       
                                                                       #cv=10 means 9 pe train hoga and 1 pe test
                                                                       #and 10 times repeat hogi ye cheez.
    print("Cross-validation scores for each fold:", scores)
    print("Average cross-validation accuracy:", scores.mean())
    return model.fit(X, y)  # Optional: train final model on full data
# Example usage after model prediction:
def F1score(model,X,y):
    y_pred = model.predict(X)
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred))

# VISUALIZATION
def visualize_data(model, dataset, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Feature Importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.barh(range(X.shape[1]), importances[indices], align="center")
    plt.yticks(range(X.shape[1]), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    # plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    # plt.show()

    # Pregnancy vs Outcome
    sns.boxplot(x="Outcome", y="Pregnancies", data=dataset, palette="pastel", hue="Outcome")
    plt.title("Pregnancy-Diabetes Relation")
    # plt.show()

    # Glucose vs Insulin
    sns.jointplot(data=dataset, x="Glucose", y="Insulin", kind="scatter", hue="Outcome", height=8)
    plt.suptitle("Glucose-Insulin Impact", y=1.02)
    # plt.show()

    # BMI vs SkinThickness
    sns.scatterplot(x="BMI", y="SkinThickness", data=dataset, hue="Outcome", palette="colorblind")
    plt.title("Skin Thickness-BMI Relation")
    # plt.show()

def main():
    # File paths
    current_dir = os.getcwd()
    input_file = os.path.join(current_dir, "DIABETES/DIABETES-PREDICTION-RANDOMFOREST/diabetes.csv")
    output_file = os.path.join(current_dir, "DIABETES/DIABETES-PREDICTION-RANDOMFOREST/Cleaned_diabetes.csv")

    # Load and clean data
    dataset = load_dataset(input_file)
    if dataset is None:
        return
    print("Before cleaning operations:\n", dataset.describe())
    dataset = cleaning_dataset(dataset)
    print("After cleaning operations:\n", dataset.describe())

    # Save cleaned data
    dataset.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to {output_file}")

    # Prepare features and labels
    X, y = prepare_features_labels(dataset)

    # Cross-validation
    model = cross_validate_model(X, y)

    # Visualization
    visualize_data(model, dataset, X)

    #lamda
    F1score(model,X,y)

if __name__ == "__main__":
    main()
