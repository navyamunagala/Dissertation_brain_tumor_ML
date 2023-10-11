import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("brain_tumor_dataset_final.csv")

print(data.head())

x = data[["AffectedArea", "severity", "Age", "Treatment", "Size"]]
y = data["Class"]

x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.4)

logModel = LogisticRegression(max_iter=500)
logModel.fit(x,y)
LogisticRegression(max_iter=500)

#Check Accuracy
print(f'Accuracy - : {logModel.score(x,y):.5f}')
pickle.dump(logModel, open("Logistic regression model.pkl", "wb"))
