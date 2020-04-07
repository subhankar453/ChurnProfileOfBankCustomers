#Part-1 Data Preprocessing

#Step-1 Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Step-2 Importing dataset and finding the input and output variables
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:12].values
y = dataset.iloc[:, 13].values

#Step-3 Encoding the categorical data in the dataset for ANN
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X1 = LabelEncoder()
labelEncoder_X2 = LabelEncoder()
x[:, 1] = labelEncoder_X1.fit_transform(x[:, 1])
y[:, 2] = labelEncoder_X2.fit_transform(x[:, 2])
oneHotEncoder = OneHotEncoder(categorical_features = [1])
x = oneHotEncoder.fit_transform(x).toarray()
x = x[:, 1:]

#Step-4 Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Step-5 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


