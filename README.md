# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import the required libraries such as pandas, numpy, and sklearn.
3. Load the dataset spam.csv using pandas.
4. Select the required columns (label and message).
5. Convert the labels spam → 1 and ham → 0.
6. Split the dataset into training data and testing data.
7. Convert the text messages into numerical form using TF-IDF Vectorizer.
8. Train the SVM (Support Vector Machine) classifier using the training data.
9. Predict whether the email is spam or not spam using the test data.
10. Calculate the accuracy of the model.
11. Display the output results.
12. Stop the program.

## Program:
```
/*


Program to implement the SVM For Spam Mail Detection..
Developed by: ARSHA JITH S J
RegisterNumber: 212224220010


*/
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("C:/Users/acer/Downloads/spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

data['label'] = data['label'].map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = SVC(kernel='linear')

model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
```

## Output:
![SVM For Spam Mail Detection](sam.png)

<img width="548" height="224" alt="Screenshot 2026-03-09 103233" src="https://github.com/user-attachments/assets/b40b9d12-c4d2-4041-b4ab-3f0efa872038" />



## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
