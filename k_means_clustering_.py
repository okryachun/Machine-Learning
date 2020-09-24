#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset as a dataframe
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

#Elbow method to determine best number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Display  the elbow graph
plt.plot(range(1,11), wcss, color='blue')
plt.title('Elbow Method Results')
plt.xlabel('K Value')
plt.ylabel('WCSS Score')
plt.show()

#Create K-means model based off of the elbow method
kmeans = KMeans(n_clusters = 5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
print(kmeans.cluster_centers_.shape)

# Plot the classification model
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c='red', label='Cluster 1' )
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c='blue', label='Cluster 2' )
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c='green', label='Cluster 3' )
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c='cyan', label='Cluster 4' )
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c='magenta', label='Cluster 5' )
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('K-Means Scatter Plot')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()