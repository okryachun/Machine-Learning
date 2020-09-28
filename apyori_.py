#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data into a dataframe
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []

#Convert dataframe into a list of lists
for i in range(0,7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])

#Create and train Apriori Model
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 21/7501,
                min_confidence = 0.2, min_lift = 3, min_length = 2,
                max_length = 2)

#Visualizing the results
results = list(rules)

#convert list into a dataframe
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

# Convert the lists into a data frame of correlating purchases
results_df = pd.DataFrame(inspect(results),
                    columns = ['Left Hand Side',
                    'Right Hand Side', 'Support', 'Confidence', 'Lift'])

#sort the dataframe rows in assending order
results_df = results_df.nlargest(n=10, columns='Lift')
