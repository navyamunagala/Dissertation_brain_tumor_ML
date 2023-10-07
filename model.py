import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

data = pd.read_csv("brain_tumor_dataset.csv")

print(data.head())

x = data[["AffectedArea", "severity", "Age", "Treatment", "Size"]]
y = data["Class"]

x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2)

model = GaussianNB()

model.fit(x_train1, y_train1)

pickle.dump(model, open("brainTumorDataset.pkl", "wb"))
