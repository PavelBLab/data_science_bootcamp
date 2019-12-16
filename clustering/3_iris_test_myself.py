import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../csv_files/iris-dataset.csv')
print(df.head())

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.scatter(df['sepal_length'], df['sepal_width'])
# ax1.set_xlabel('sepal_length')
# ax1.set_ylabel('sepal_width')
# ax2.scatter(df['petal_length'], df['petal_width'])
# ax2.set_xlabel('petal_length')
# ax2.set_ylabel('petal_width')


''' Clustering (unscaled data) '''
df_copy = df.copy()
kmeans = KMeans(3)
kmeans.fit(df_copy)
df_copy['cluster_predict'] = kmeans.fit_predict(df_copy)

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.scatter(df_copy['sepal_length'], df_copy['sepal_width'], c=df_copy['cluster_predict'], cmap='rainbow')
# ax1.set_xlabel('sepal_length')
# ax1.set_ylabel('sepal_width')
# ax2.scatter(df_copy['petal_length'], df_copy['petal_width'], c=df_copy['cluster_predict'], cmap='rainbow')
# ax2.set_xlabel('petal_length')
# ax2.set_ylabel('petal_width')
# plt.title('Unscaled data')


''' Standardize the variable '''
from sklearn import preprocessing

scaled = preprocessing.scale(df)
# print(scaled)


''' Clustering (scaled data) '''
df_scaled = df.copy()
kmeans_scale = KMeans(3)
kmeans.fit(scaled)
df_scaled['cluster_predict'] = kmeans_scale.fit_predict(scaled)

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.scatter(df_scaled['sepal_length'], df_scaled['sepal_width'], c=df_scaled['cluster_predict'], cmap='rainbow')
# ax1.set_xlabel('sepal_length')
# ax1.set_ylabel('sepal_width')
# ax2.scatter(df_scaled['petal_length'], df_scaled['petal_width'], c=df_scaled['cluster_predict'], cmap='rainbow')
# ax2.set_xlabel('petal_length')
# ax2.set_ylabel('petal_width')
# plt.title('Scaled data')

''' Take Advantage of the Elbow Method '''
wcss = []
cluster_num = 10
for i in range(1, cluster_num):
    kmeans = KMeans(i)
    kmeans.fit(scaled)
    wcss.append(kmeans.inertia_)
# plt.plot(range(1, cluster_num), wcss)





real_data = pd.read_csv('../csv_files/iris-with-answers.csv')
# print(real_data.head())
# print(real_data['species'].unique())
real_data['species_clusters'] = real_data['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

cmap = 'tab10'
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.scatter(df_copy['sepal_length'], df_copy['sepal_width'], c=df_copy['cluster_predict'], cmap=cmap)
ax1.set_xlabel('sepal_length_unscaled')
ax1.set_ylabel('sepal_width')
ax2.scatter(df_scaled['sepal_length'], df_scaled['sepal_width'], c=df_scaled['cluster_predict'], cmap=cmap)
ax2.set_xlabel('sepal_length_scaled')
ax3.scatter(real_data['sepal_length'], real_data['sepal_width'], c=real_data['species_clusters'], cmap=cmap)
ax3.set_xlabel('sepal_length_real')


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.scatter(df_copy['petal_length'], df_copy['petal_width'], c=df_copy['cluster_predict'], cmap=cmap)
ax1.set_xlabel('petal_length_unscaled')
ax1.set_ylabel('petal_width')
ax2.scatter(df_scaled['petal_length'], df_scaled['petal_width'], c=df_scaled['cluster_predict'], cmap=cmap)
ax2.set_xlabel('petal_length_scaled')
ax3.scatter(real_data['petal_length'], real_data['petal_width'], c=real_data['species_clusters'], cmap=cmap)
ax3.set_xlabel('petal_length_real')

fig.title('fffffffffffffffffffff')
plt.show()

