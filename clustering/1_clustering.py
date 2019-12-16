import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../csv_files/3.12. Example.csv')
print(df.head())
# print(df.describe())

# plt.scatter(df['Satisfaction'], df['Loyalty'])
# plt.xlabel('Satisfaction')
# plt.ylabel('Loyalty')
# plt.show()

df_copy = df.copy()

kmeans = KMeans(2)
print(kmeans.fit(df_copy))

df_with_cluster = df_copy.copy()
df_with_cluster['cluster_pred'] = kmeans.fit_predict(df_copy)
# print(df_with_cluster.head())

# plt.scatter(df['Satisfaction'], df['Loyalty'], c=df_copy['cluster_pred'], cmap='rainbow')
# plt.xlabel('Satisfaction')
# plt.ylabel('Loyalty')
# plt.show()



'''#########################################################################################'''
''' Preprocessing '''
df_scaled = preprocessing.scale(df)
# print(df_scaled)

''' Take advantage from Elbow method '''
wcss = []
for i in range(1, 10):
    kmeans = KMeans(i)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)
# print(wcss)
# plt.plot(range(1, 10), wcss)
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')

kmeans_new = KMeans(4)
df_copy['cluster_pred'] = kmeans_new.fit_predict(df_scaled)
# print(df_copy.head())
plt.scatter(df_copy['Satisfaction'], df_copy['Loyalty'], c=df_copy['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()
'''#########################################################################################'''











