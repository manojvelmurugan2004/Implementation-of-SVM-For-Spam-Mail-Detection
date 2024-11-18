# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 : Gather a labeled dataset containing both spam and non-spam emails, with labels typically as 0 (non-spam) and 1 (spam)
STEP 2 : Clean the email text by removing punctuation, converting text to lowercase, and removing stop words to reduce noise.
STEP 3 : Split the dataset into training and testing sets, typically using an 80-20 or 70-30 ratio.
STEP 4 : Train the SVM on the training data, allowing it to learn patterns associated with spam and non-spam emails.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by : MANOJ MV 
RegisterNumber : 212222220023  
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
df=pd.read_csv("spam.csv",encoding='Windows-1252')

df.head()

df.info()

df.isnull().sum()

x=df['v2'].values
y=df['v1'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
#CountVectorizer is convert text into numerical data

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![384794935-2b903aa2-7226-42c2-afae-03b12ebcbc65](https://github.com/user-attachments/assets/7b669ae4-92c4-494a-9980-e1e4dea98200)
![384794977-5d6c9720-b536-496b-9d8d-8c201dab4002](https://github.com/user-attachments/assets/38c59dad-89fb-4619-8f1d-5d63086550bf)
![384795025-d6994dbd-4c5f-4df5-8845-99a68656b101](https://github.com/user-attachments/assets/7b1bb370-c08e-4a25-b9a8-5ee3f1dc8cfd)
## Accuracy
![384795075-dba52b05-04e6-4fd6-b7d9-c965d651cece](https://github.com/user-attachments/assets/2082d7d7-5821-4421-91bc-90bd21446452)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
