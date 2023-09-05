# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Adhithya M R
RegisterNumber:  212222240002
*/
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data

plt.scatter(x_train,y_train,color="black") 
plt.plot(x_train,regressor.predict(x_train),color="red") 
plt.title("Hours VS scores (learning set)") 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)




```

## Output:
### df.head():



![263445760-4a10342a-4fcc-47db-a298-4d87d6485991](https://github.com/AdhithyaMR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118834761/146d35ed-4f14-4c8e-bdd3-07ade7e5b9d3)

### df.tail():


![263445796-5b1b966e-600d-4aec-821c-0df2d9bbc311](https://github.com/AdhithyaMR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118834761/7af9b653-2f78-4548-8454-8063b529eeb9)

### Array value of X:

![263445955-bfc57abc-2843-49c2-a296-0ea9c2a26bba](https://github.com/AdhithyaMR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118834761/59af3702-f2a2-451a-9b21-e2b71b6d7e9d)

### Array value of Y:
![263445972-aadb93a4-2245-4963-9b6a-a1e83a4feaea](https://github.com/AdhithyaMR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118834761/dd615872-8acf-4558-906c-b284e32853a2)

### Values of Y prediction:
![263445996-f5e5cf9a-c40c-40c7-bff6-21e7ac15f965](https://github.com/AdhithyaMR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118834761/47605b32-5c9a-41dc-b626-7b6cb241ece0)

### Values of Y test:
![263446021-37b5fb12-d11a-48bc-8792-1798f55b3876](https://github.com/AdhithyaMR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118834761/c40f0ef1-171a-47e2-b469-d8ba6a1d868a)

### Training Set Graph:



![263446040-60125d6d-4c88-4724-9924-a3f66bab0699](https://github.com/AdhithyaMR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118834761/9523a6eb-3cb5-46d1-b807-d95834a8ce3e)

### Test Set Graph:
![263446057-c3e2fb4c-0f13-47a4-af45-b929f1ca90d3](https://github.com/AdhithyaMR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118834761/e858135a-132b-4260-805c-d37690ffb7df)

### Values of MSE,MAE and RMSE:


![263446098-513a0073-7dd8-427e-b250-4415a60ea7a1](https://github.com/AdhithyaMR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118834761/e46bf57a-0ce0-41b9-ba87-8126ae430f28)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
