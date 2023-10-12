import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

data = pd.read_csv("predict_treatment.csv")

print(data.head())

x = data[["AffectedArea", "severity", "Age", "Size", "Tumortype"]]
y = data["Class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

pickle.dump(model, open("predictTreatment.pickle", "wb"))
