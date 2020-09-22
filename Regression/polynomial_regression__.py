# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Create a linear regression model for a future comparison
from sklearn.linear_model import LinearRegression
linreg = LinearRegression();
linreg.fit(X, y)

# polynomial regression setup
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Create a new linear regression model to use for the polynomial model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# plot linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linreg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# plot polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X,lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# plotting a smoother graph, breaking down into smaller intervals
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Use the model to make a prediction, first have to transform the value using 
# PolynomialFeatures transformer
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
