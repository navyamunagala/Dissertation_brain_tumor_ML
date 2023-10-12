import inline as inline
import matplotlib
import pandas as pd
import seaborn as sns
from imblearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

sns.set(color_codes=True)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier


dataset = pd.read_csv("brain_tumor_dataset.csv")

print("---------------------1.First few rows of the dataset--------------------")
print(dataset.head())

print("----------------------2.Last few rows of the dataset--------------------")
print(dataset.tail(2))

print("----------------------3. dataset information-----------------------------")
print(dataset.info())

print("----------------------4. dataset shape--------------------------------")
print(dataset.shape)

print("---------------------5. columns present in dataset----------------------")
print(dataset.columns)

print("---------------------6. further info of dataset-------------------------")
print(dataset.describe())

# countplt, ax = plt.subplots(figsize = (10,5))
# ax = sns.countplot(x = 'Class', data = dataset)
# print(dataset.ax)

print("------------------------7. histogram of dataset---------------------")
print(dataset.hist())

print("-----------------------8.pair plot shows the relationship between the attributes--------")
print(sns.pairplot(dataset, hue="Class"))

data = dataset.drop(["Class"], axis=1)
print(data)
print("-----------------------------9. correlation matrix-----------------")
print(data.corr())

print("-------------------10. heat map--------------------------------")
fig = plt.figure(num=None, figsize=(10, 10))
print(sns.heatmap(data, annot=True, fmt='.2f'))

print("---------------------11.Data Pre-processing and Cleaning-------------------------------------------")
# print(dataset.head())
print(dataset.duplicated())

unique_sets = dataset.drop_duplicates(
    subset=["Patient ID", "AffectedArea", "severity", "Age", "Size", "Tumortype", "Class"])
print(unique_sets)

print(unique_sets.shape)

print(unique_sets.isnull().values.any())

data = unique_sets[["Patient ID", "AffectedArea", "severity", "Age", "Size", "Tumortype", "Class"]].replace("empty",
                                                                                                            np.nan)
print(data.head(10))

print(data.describe())

print(data['Patient ID'].fillna(data['Patient ID'].median(), inplace=True),
      data['Age'].fillna(data['Age'].median(), inplace=True),
      data['Size'].fillna(data['Size'].median(), inplace=True),
      data['Tumortype'].fillna(data['Tumortype'].median(), inplace=True),
      data['severity'].fillna(data['severity'].median(), inplace=True),
      )

print(data.head())

# -------------------------------12.remove outliers----------------------

# Splitting the area of the dataset into 4 bins of tumor sizes and applying discretization to transform the continuos data into discrete values by grouping it.

print("----------------------------13. split data into 4 bins-----------------------------------")
data["size_data"] = pd.cut(x=data["Size"], bins=4, labels=["Tiny", "Small", "Medium", "Big"])
print(data['size_data'].value_counts())

print("----------------------------------Method to draw bar plot for the data set------------------------")


def draw_barplot(x):
    s = x.value_counts()
    plt.bar(s.index, s.values)


size_dat = draw_barplot(data["size_data"])
print(size_dat)

print(
    "-----------------------------Applying feature engineering technique - Discretization to tranform continuos numerical---")

discretizer = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="uniform")
data["Size_uniform_discrete"] = discretizer.fit_transform(data["Size"].values.reshape(-1, 1)).astype(int)
print(data.head())

print(discretizer.bin_edges_)

print(
    "---------------------Applying Equal frequency discretization where data points will be same but the width might differ------")

discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
data['Size_eq_freq'] = discretizer.fit_transform(data['Size'].values.reshape(-1, 1)).astype(int)
print(data.head())

print(discretizer.bin_edges_)

print(
    "----------------------Applying Cluster algorithm to convert continous variable to discrete variable------------------")
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
data['Size_kmeans_discrete'] = discretizer.fit_transform(data['Size'].values.reshape(-1, 1)).astype(int)
print(data.head())

clean_data_set = data.drop(['Class', 'size_data'], axis=1)
print(clean_data_set)

print("----------------------------------normalize data---------------------------")
Scaler = MinMaxScaler()
df_normalised = Scaler.fit_transform(clean_data_set)
print(df_normalised)
print("-------------------- clean data set--------------------------------")
print(clean_data_set)

print("-------------------------normalized data-------------------")
allColumns = ['Patient ID', 'AffectedArea', 'severity', 'Age', 'Size', 'Tumortype', 'Size_uniform_discrete',
              'Size_eq_freq', 'Size_kmeans_discrete']

df_normalised = pd.DataFrame(df_normalised, columns=allColumns)
# df_normalised.drop(['size_data','Size_uniform_discrete','Size_eq_freq'], axis = 1, inplace = True)
print(df_normalised.shape)
filteredDataSet = df_normalised.join(data.filter(['Class'], axis=1))
print(filteredDataSet)

print(sns.pairplot(filteredDataSet))

print("--------------------filtered dataset details-----------------------")
print(filteredDataSet.describe())

print(df_normalised)

print("-----------------model building---------------------------")
print("---------print X details---------------")
X = filteredDataSet.drop(["Class"], axis=1)
print(X)
print("---------print Y details---------------")
Y = filteredDataSet["Class"]
print(Y)

print("------------------------Split X and y into training and test set in 80:20 ratio----------------")
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.2, random_state=1)
columns1 = X_train1.columns
print("Train-Set-1 :" + str(X_train1.shape))
print("Test-Set-1:" + str(X_test1.shape))

print("-----------------Split X and y into training and test set in 10:90 ratio-----------")
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.9, random_state=1)
columns2 = X_train2.columns
print("Train-Set-2 :" + str(X_train2.shape))
print("Test-Set-2:" + str(X_test2.shape))

print("--------------------scaling---------------")
scaler = StandardScaler()
scaler.fit(X_train1)
X_train1 = scaler.transform(X_train1)
X_test1 = scaler.transform(X_test1)
print(X_train1)
print(X_test1)

print("------now use SMOTE to oversample our train data which have features data_train_X and labels in data_train_y---")

Ov_sampling = SMOTE(random_state=0)

ov_data_X, ov_data_Y = Ov_sampling.fit_resample = (X_train1, Y_train1)
ov_data_X = pd.DataFrame(data=ov_data_X, columns=columns1)
ov_data_Y = pd.DataFrame(ov_data_Y, columns=['Class'])
print("Case 1: Train = 80% , Test = 20%")
print('length of oversampled data is   ', len(ov_data_X))
print('Number of no subscription in oversampled data ', len(ov_data_Y[ov_data_Y['Class'] == 0]))
print('Number of subscription ', len(ov_data_Y[ov_data_Y['Class'] == 1]))
print('Proportion of no subscription data in oversampled data is ',
      len(ov_data_Y[ov_data_Y['Class'] == 0]) / len(ov_data_X))
print('Proportion of subscription data in oversampled data is ',
      len(ov_data_Y[ov_data_Y['Class'] == 1]) / len(ov_data_X))
print("\n\n")

print("-------------**Naive** Bayes Model----------------")
model=GaussianNB()
model.fit(ov_data_X,ov_data_Y)
pred = model.predict(X_test1)

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test1, pred)*100)

print("---------------Logistic regression-------------------------")
logModel = LogisticRegression(max_iter=2500)
logModel.fit(X,Y)
LogisticRegression(max_iter=2500)

#Check Accuracy
print(f'Accuracy - : {logModel.score(X,Y):.3f}')

# print("-------------K-fold cross validation-------------------")
# # Create an instance of Pipeline
# pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=1000, max_depth=10))
# # Pass instance of pipeline and training and test data set
# # cv=5 represents the StratifiedKFold with 5 folds
# #
# scores = cross_val_score(pipeline, X=X_train1, Y=Y_train1, cv=5, n_jobs=1)
#
# print('Cross Validation accuracy scores: %s' % scores)
#
# print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

print("----------------decision tree-------")
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.2, random_state=1)
model = DecisionTreeClassifier()
model.fit(X_train1,Y_train1)
pred = model.predict(X_test1)
print("DT model accuracy(in %):", metrics.accuracy_score(Y_test1, pred)*100)
# print("DT model accuracy(in %):", metrics.accuracy_score(X_test1, pred)*100)

# print("----------------Reguralization------------------------")
# clf = LogisticRegression(random_state=0).fit(X,Y)
# clf.score(X,Y)
#
# print(clf.predict(X[:2, :]))
# # print(clf.predict_proba(X[:2, :]))
# # print(clf.score(X, Y))

