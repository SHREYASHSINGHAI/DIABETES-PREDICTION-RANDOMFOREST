import pandas as pd


#GETTING THE DATASET
df=pd.read_csv(r"DIABETES/DIABETES-PREDICTION-RANDOMFOREST/diabetes.csv")
dataset=pd.DataFrame(df)
print(dataset.describe())
