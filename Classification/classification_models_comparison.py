#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

#import dataset as a dataframe
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#feature scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classification_types = ['Logistic Regression', 'KNN',
                        'SVM linear', 'SVM non-linear',
                        'Naive Bayes', 'Decision Tree',
                        'Random Forest']
cm = []
score = []

#Logistic Regression Classification model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm.append(confusion_matrix(y_test, y_pred))
score.append(accuracy_score(y_test, y_pred))

#KNN Classification
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm.append(confusion_matrix(y_test, y_pred))
score.append(accuracy_score(y_test, y_pred))

#SVM Linear classifier
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm.append(confusion_matrix(y_test, y_pred))
score.append(accuracy_score(y_test, y_pred))

#SVM Linear classifier
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm.append(confusion_matrix(y_test, y_pred))
score.append(accuracy_score(y_test, y_pred))

#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm.append(confusion_matrix(y_test, y_pred))
score.append(accuracy_score(y_test, y_pred))

#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm.append(confusion_matrix(y_test, y_pred))
score.append(accuracy_score(y_test, y_pred))

#Random Forest Classification model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm.append(confusion_matrix(y_test, y_pred))
score.append(accuracy_score(y_test, y_pred))

for i in range(len(classification_types)):
    print(classification_types[i])
    print(cm[i])
    print(score[i])
    print('\n\n')