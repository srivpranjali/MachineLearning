## Problem Statement:
## Predict the gender of a person from the specified data sets of height and weight
## Dataset used is https://www.kaggle.com/shawon10/male-female-detection-by-height-weight-knn/data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

## create data frame 
data_set = pd.read_csv("weight-height.csv")

## Labelencoding the gender column as 0 or 1
le = LabelEncoder()
data_set['Gender']=le.fit_transform(data_set['Gender'])

## Creating arrays for x and y data pointers
x = data_set.iloc[:,[0,1]].values
y = data_set.iloc[:,[2]].values

## test data point for prediction
x1 = data_set.iloc[-1:,[0,1]].values
print(x1)

## creating training and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

## feature scaling to standard
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.fit_transform(x_test)
x1_test = st_x.fit_transform(x1)

## implementing K-NN algorithm
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train,y_train)

## prediction
y_pred = classifier.predict(x_test)
y1_pred = classifier.predict(x1_test)
gen = y1_pred[0]
if gen == 1:
    print("Person is male")
else:
    print("Person is female")





