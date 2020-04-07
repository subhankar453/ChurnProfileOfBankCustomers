#Part-1 Data Preprocessing

#Step-1 Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Step-2 Importing dataset and finding the input and output variables
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Step-3 Encoding the categorical data in the dataset for ANN
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x);
x = x[:, 1:]
from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])

#Step-4 Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Step-5 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#Part-2 Building the ANN network

#Step-1 Importing keras library. Then Sequential function is imported to initialize the ANN model
#The Dense function is imported to create the layers in ANN network
import keras
from keras.models import Sequential
from keras.layers import Dense

#Step-2 Developing the network layers
classifier = Sequential()
#Creating the input layer and the first hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='uniform', input_dim = 11))
#Creating the second hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='uniform'))
#Creating the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer='uniform'))
#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', weighted_metrics = ['accuracy'])
#Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

