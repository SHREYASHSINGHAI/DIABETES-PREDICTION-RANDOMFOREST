import pandas as pd
import os
from sklearn.impute import SimpleImputer  #DATA CLEANING AND MISSING VALUE HANDLING
import numpy as np  #DATA CLEANING AND MISSING VALUE HANDLING
from sklearn.model_selection import train_test_split  #SPLITTING THE DATASET
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt #VISUALIZATION
import seaborn as sns


#GETTING THE DATASET
def load_dataset(file_path):
    try:
        dataset=pd.read_csv(file_path)
        print("Dataset successfully loaded!")
        return dataset
    except FileNotFoundError:
        print(f"Error file not found at {file_path}")
        return None


#DATA CLEANING AND MISSING VALUE HANDLING   
def cleaning_dataset(dataset):
    columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    imputer=SimpleImputer(missing_values=0,strategy="median")
    dataset[columns_to_fix]=imputer.fit_transform(dataset[columns_to_fix])
    dataset.to_csv(r"DIABETES\DIABETES-PREDICTION-RANDOMFOREST\Cleaned_diabetes.csv",index=False)
    return dataset


#DEVIDING AND SPLITTING THE DATASET
def split_dataset(dataset):
    X=dataset[["Pregnancies","Glucose","BloodPressure","SkinThickness",
              "Insulin","BMI","DiabetesPedigreeFunction","Age"]]
    y=dataset["Outcome"]
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    print("Number of x_train values : ",x_train.shape[0])
    print("Number of x_test values : ",x_test.shape[0])
    print(" ")
    #checking if any null value exists
    print("The dependent variable contains following null values : ",X.isnull().sum())
    print(" ")
    return x_train,x_test,y_train,y_test


#TRAINING MODEL WITH RANDOMFOREST
def train_model(x_train,y_train):
    model=RandomForestClassifier(n_estimators=160,max_features=None)
    model.fit(x_train,y_train)
    print("Model training completed!")
    return model

#MODEL EVALUATION
def evaluate_model(model,x_test,y_test):
    prediction=model.predict(x_test)
    accuracy=accuracy_score(y_test,prediction)
    print(f"The accuracy of the model is : {accuracy}")
    return accuracy,prediction


#VISUALIZING THROUGH GRAPHS
def visualize_data(model,dataset,X):   
    # Extracting feature importances
    importances = model.feature_importances_#it gives score how important a feature is in making predictions.
    indices = np.argsort(importances)[::-1]# Sorting the feature importances in descending order

    # Plotting the feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.barh(range(X.shape[1]), importances[indices], align="center")
    plt.yticks(range(X.shape[1]), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.show()

    #Features correlation
    features=["Glucose","Insulin","BMI","SkinThickness","BloodPressure","Outcome"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm',fmt=".2f")
    plt.title("Feature Correlation Heatmap") #
    plt.show()

    #Pregnancy Diabetes Relation
    sns.boxplot(x="Outcome", y="Pregnancies", data=dataset,palette="pastel",hue="Outcome")
    plt.title("Pregnancy-Diabetes Relation") 
    plt.show()

    #Glucose Insulin Relation
    sns.jointplot(data=dataset, x="Glucose", y="Insulin", kind="scatter",hue="Outcome", height=8)
    plt.suptitle("Glucpse-Insulin impact")
    plt.show()

    #Skin thickness and BMI relation
    sns.scatterplot(x="BMI", y="SkinThickness", data=dataset, hue="Outcome", palette="colorblind")
    plt.title("Skin Thickness-BMI Relation")
    plt.show()

def main():
# Define file paths
    current_dir = os.getcwd()
    input_file = os.path.join(current_dir, "DIABETES/DIABETES-PREDICTION-RANDOMFOREST/diabetes.csv")
    output_file = os.path.join(current_dir, "DIABETES/DIABETES-PREDICTION-RANDOMFOREST/Cleaned_diabetes.csv")

# Load dataset
    dataset = load_dataset(input_file)
    if dataset is None:
        return
    print("Before cleaning operations:")
    print(dataset.describe())

# Clean data
    dataset = cleaning_dataset(dataset)
    print("After cleaning operations:")
    print(dataset.describe())


# Save cleaned data
    dataset.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to {output_file}")

# Split dataset
    x_train, x_test, y_train, y_test = split_dataset(dataset)

# Train model
    model = train_model(x_train, y_train)

 # Evaluate model
    evaluate_model(model, x_test, y_test)

 # Visualize data
    visualize_data(model,dataset, x_train)

if __name__ == "__main__":
    main()