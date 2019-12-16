import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
import os.path as path

csv_folder = path.abspath(path.join(path.join(__file__, "../../.."), 'csv_files'))
# print(path.abspath(__file__))  os.path.abspath(__file__) method in Python is used to get the file path and 2nd argument ../../.. move 3 levels up
# print(csv_folder)

df = pd.read_csv(path.join(csv_folder, '1.01. Simple linear regression.csv'))

print(df.head(2))
print(list(df.columns))

x = df.loc[:, 'SAT']
y = df.loc[:, 'GPA']
print(x.shape)
print(y.shape)

x_matrix = x.values.reshape(-1, 1)
# print(x_matrix)
print(x_matrix.shape)

reg = LinearRegression()  # reg is an instance of LinearRegression class
reg.fit(x_matrix, y)
print(reg.fit(x_matrix, y))

pearson_r_score = reg.score(x_matrix, y)
print(pearson_r_score)

b1_coef = reg.coef_
print(b1_coef)

b0_intercept = reg.intercept_
print(b0_intercept)

X = pd.DataFrame(data=[1740])
# X = np.array([1740])
# print(type(X))
print(reg.predict(X))

new_data = pd.DataFrame(data=[1740, 1760], columns=['SAT'])
print(new_data)
print(reg.predict(new_data))

new_data['Prediction_GPA'] = reg.predict(new_data)
print(new_data)

sns.regplot(x, y, label=['SAT', 'GPA'])
yhat = b0_intercept + b1_coef * x_matrix
# print(yhat)
plt.plot(x, yhat, linewidth=2, label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.legend()
plt.show()






