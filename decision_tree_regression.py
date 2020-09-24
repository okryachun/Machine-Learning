# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# imort Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor as DTR
regressor = DTR()
regressor.fit(X, y)

# Check prediction value
print(regressor.predict([[6.5]]))

#plot the decision tree graph
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Decision Tree Regressor Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#print the smooth graph
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regressor Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()