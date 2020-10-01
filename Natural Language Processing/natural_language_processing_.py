#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#import data
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter ='\t', quoting = 3)

#Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


corpus = []
for i in range(0, len(dataset)):
    #remove anything that isn't a letter and replace with a space
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower() #convert all to lower case
    review = review.split() #split the words into individual lists
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    all_stopwords.remove('isn')
    all_stopwords.remove('doesn')
    #iterate through all words, if not a stop word, then perform stemming
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review) #join all words with spaces in between
    corpus.append(review) #store into corpus list

#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#Split data into training and test sets
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.25)

#***************************************************************
# USE different Classification Models to determine the best fit.
# **************************************************************

class_types = ['Naive', 'Logistic', 'KNN', 'SVC Linear', 'SVC Non-Linear',
               'Decision Tree', 'Random Forest']
y_pred = []
#Create Niaive Bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred.append(classifier.predict(X_test))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred.append(classifier.predict(X_test))

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
y_pred.append(classifier.predict(X_test))

#SVC Linear
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)
y_pred.append(classifier.predict(X_test))

#SVC Non-Linear
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)
y_pred.append(classifier.predict(X_test))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred.append(classifier.predict(X_test))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)
y_pred.append(classifier.predict(X_test))


#Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
for i in range(0, len(class_types)):
    print(class_types[i])
    print(confusion_matrix(y_test, y_pred[i]))
    print(accuracy_score(y_test, y_pred[i]))
