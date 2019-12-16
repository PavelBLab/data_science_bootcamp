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

df = pd.read_csv(path.join(csv_folder, 'real_estate_price_size.csv'))

print(df.head(2))
print(list(df.columns))


x = df.loc[:, 'size']
X = x.values.reshape(-1, 1)
y = df.loc[:, 'price']
print(X.shape, y.shape)

reg = LinearRegression()
reg.fit(X, y)

r_score = reg.score(X, y)
print(r_score)

b0 = reg.intercept_
b1 = reg.coef_
print(b0, b1)


new_data = pd.DataFrame(data=[700, 800], columns=['size'])
price_prediction = reg.predict(new_data)
print(price_prediction)


sns.regplot(x, y, label=['size', 'price'])
plt.show()




















