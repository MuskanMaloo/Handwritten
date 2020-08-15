import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys
import numpy as np
import pandas as pd
import cv2
os.chdir("C:/New folder")
df=pd.read_csv("mnist_train.csv",nrows=60000)
x=df.drop(['label'],axis=1).values
y=df['label'].values
def x_matrics(df):
    x=df.drop(['label'],axis=1).values
    y=df['label'].values
    return x,y
    
digit=x[5]
somedigit=digit.reshape(28,28)
plt.imshow( somedigit)
plt.show()
train,test=train_test_split(df,test_size=0.2,random_state=2)
x_train,y_train=x_matrics(train)
x_test,y_test=x_matrics(test)
rand_model=RandomForestClassifier()
rand_model.fit(x_train,y_train)
predict=rand_model.predict(x_test)
score=accuracy_score(y_test,predict)
print(score)

