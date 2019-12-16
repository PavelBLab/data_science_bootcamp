import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import os.path as path

csv_folder = path.abspath(path.join(path.join(__file__, "../../.."), 'csv_files'))
# print(path.abspath(__file__))  os.path.abspath(__file__) method in Python is used to get the file path and 2nd argument ../../.. move 3 levels up
# print(csv_folder)

df = pd.read_csv(path.join(csv_folder, 'real_estate_price_size_year.csv'))

# print(df.head(2))
# print(list(df.columns))

x = df.loc[:, ['size', 'year']]
y = df.loc[:, 'price']

reg = LinearRegression()
reg.fit(x, y)

print('intercept ==>', reg.intercept_, '\ncoefficient ==>', reg.coef_, '\npearson r score ==>', reg.score(x, y))

f_statistic_and_p_value = f_regression(x, y)
summary_statistic_table = pd.DataFrame(data=x.columns, columns=['Features'])
summary_statistic_table['Coefficients'] = reg.coef_
summary_statistic_table['F-Statistic'] = f_statistic_and_p_value[0]
summary_statistic_table['P-Value'] = f_statistic_and_p_value[1].round(3)
print(summary_statistic_table)


sns.regplot(x['size'], y)
plt.show()










