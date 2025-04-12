import pandas as pd
from sklearn.impute import SimpleImputer  #DATA CLEANING AND MISSING VALUE HANDLING
import numpy as np  #DATA CLEANING AND MISSING VALUE HANDLING
from sklearn.model_selection import train_test_split  #SPLITTING THE DATASET
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#GETTING THE DATASET
df=pd.read_csv(r"DIABETES/DIABETES-PREDICTION-RANDOMFOREST/diabetes.csv")
dataset=pd.DataFrame(df)
print("Before cleaning operations : ")
print(dataset.describe())
print("  ")


#DATA CLEANING AND MISSING VALUE HANDLING
#replacing 0 with medians
columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
imputer=SimpleImputer(missing_values=0,strategy="median")
dataset[columns_to_fix]=imputer.fit_transform(dataset[columns_to_fix])
print("After cleaning operations : ")
print(dataset.describe())
print(" ")
dataset.to_csv(r"D:\college\DIABETES\DIABETES-PREDICTION-RANDOMFOREST\Cleaned_diabetes.csv",index=False)


#DEVIDING DATASET
X=dataset[["Pregnancies","Glucose",
           "BloodPressure","SkinThickness",
           "Insulin","BMI",
           "DiabetesPedigreeFunction","Age"]]
y=dataset["Outcome"]


#SPLITTING THE DATASET
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Number of x_train values : ",x_train.shape[0])
print("Number of x_test values : ",x_test.shape[0])
print(" ")

print("The dependent variable contains following null values : ",X.isnull().sum())
print(" ")


#APPLYING RANDOMFOREST MODEL
model=RandomForestClassifier(n_estimators=160,max_features=None)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(prediction)


#MODEL EVALUATION
Accuracy=accuracy_score(y_test,prediction)
print("The accuracy of the model is : ",Accuracy)


#VISUALIZING THROUGH GRAPHS
import matplotlib.pyplot as plt
import seaborn as sns

features=["Glucose","Insulin","BMI","SkinThickness","BloodPressure","Outcome"]
plt.figure(figsize=(10, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap") #
plt.show()

sns.boxplot(x="Outcome", y="Pregnancies", data=dataset,palette="pastel")
plt.title("Pregnancy-Diabetes Relation") #
plt.show()

sns.jointplot(data=dataset, x="Glucose", y="Insulin", kind="scatter",hue="Outcome", height=8)
plt.title("Glucpse-Insulin impact")
plt.show()

sns.scatterplot(x="BMI", y="SkinThickness", data=dataset, hue="Outcome", palette="colorblind")
plt.title("Skin Thickness-BMI Relation")
plt.show()