import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../csv_files/iris-dataset.csv')
print(df.head())

# plt.scatter(df['sepal_length'], df['sepal_width'])
# plt.xlabel('sepal_length')
# plt.ylabel('sepal_width')



''' Clustering (unscaled data) '''
df_copy = df.copy()
kmeans = KMeans(2)
kmeans.fit(df_copy)
# create a copy of data, so we can see the clusters next to the original data
clusters = df_copy.copy()
clusters['cluster_predict'] = kmeans.fit_predict(df_copy)
# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
# plt.scatter(clusters['sepal_length'], clusters['sepal_width'], c=clusters['cluster_predict'], cmap='rainbow')
# plt.xlabel('sepal_length')
# plt.ylabel('sepal_width')

''' Standardize the variable '''
from sklearn import preprocessing
scaled = preprocessing.scale(df_copy)
# print(df_scaled)


''' Clustering (scaled data) '''
# create a k-means object with 2 clusters
kmeans_scaled = KMeans(4)
kmeans_scaled.fit(scaled)

# create a copy of data, so we can see the clusters next to the original data
clusters_scaled = df.copy()
clusters_scaled['cluster_predict'] = kmeans_scaled.fit_predict(scaled)
print(clusters_scaled.head())
# plt.scatter(clusters_scaled['sepal_length'], clusters_scaled['sepal_width'], c=clusters_scaled['cluster_predict'], cmap='rainbow')
# plt.xlabel('sepal_length')
# plt.ylabel('sepal_width')


''' Take Advantage of the Elbow Method '''
wcss = []
# 'cluster_num' is a that keeps track the highest number of clusters we want to use the WCSS method for.
# We have it set at 10 right now, but it is completely arbitrary.
cluster_num = 10
for i in range(1, cluster_num):
    kmeans = KMeans(i)
    kmeans.fit(scaled)
    wcss.append(kmeans.inertia_)
# print(wcss)

# plt.plot(range(1, cluster_num), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Within-cluster Sum of Squares')
# plt.show()


real_data = pd.read_csv('../csv_files/iris-with-answers.csv')
print(real_data.head())
print(real_data['species'].unique())
real_data['species_clusters'] = real_data['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
print(real_data.head())
# plt.scatter(real_data['sepal_length'], real_data['sepal_width'], c=real_data['species_clusters'], cmap='rainbow')
# plt.xlabel('sepal_length')
# plt.ylabel('sepal_width')
plt.scatter(real_data['petal_length'], real_data['petal_width'], c=real_data['species_clusters'], cmap='rainbow')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.show()


