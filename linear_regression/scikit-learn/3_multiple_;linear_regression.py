import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression    # this module is to find f-statistics and p-value
from sklearn.preprocessing import StandardScaler    # this modele is to do feature scaling or standartisation
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import os.path as path

csv_folder = path.abspath(path.join(path.join(__file__, "../../.."), 'csv_files'))
# print(path.abspath(__file__))  os.path.abspath(__file__) method in Python is used to get the file path and 2nd argument ../../.. move 3 levels up
# print(csv_folder)

df = pd.read_csv(path.join(csv_folder, '1.02. Multiple linear regression.csv'))

print(df.head(2))
print(list(df.columns))


x = df.loc[:, ['SAT', 'Rand 1,2,3']]
y = df.loc[:, 'GPA']

'''
Get P value through statsmodels.api
'''
x_constant = sm.add_constant(x)
summary_statistic = sm.OLS(y, x_constant).fit().summary()
print(summary_statistic)
'''=================================================='''


reg = LinearRegression()
reg.fit(x, y)
print('intercept ==>', reg.intercept_, '\ncoefficient ==>', reg.coef_, '\npearson r score ==>', reg.score(x, y))
print('Find a F-Statistic ==>', f_regression(x, y)[0], '\nFind a P value ==>', f_regression(x, y)[1].round(3))
print(x.columns)


f_statistic_and_p_value = f_regression(x, y)
reg_summary = pd.DataFrame(data=x.columns, columns=['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['F-Statistic'] = f_statistic_and_p_value[0]
reg_summary['P-Value'] = f_statistic_and_p_value[1].round(3)
print(reg_summary)

scaler = StandardScaler()
scaler.fit(x)
x_features_scaled = scaler.transform(x)
# print(scaler.fit_transform(x))

reg = LinearRegression()
reg.fit(x_features_scaled, y)

print('intercept ==>', reg.intercept_, '\ncoefficient ==>', reg.coef_, '\npearson r score ==>', reg.score(x_features_scaled, y))
f_statistic_and_p_value = f_regression(x_features_scaled, y)
summary_statistic_table = pd.DataFrame(data=x.columns, columns=['Features'])
summary_statistic_table['Coefficients'] = reg.coef_
summary_statistic_table['Intercept/Bias'] = reg.intercept_
summary_statistic_table['F-Statistic'] = f_statistic_and_p_value[0]
summary_statistic_table['P-Value'] = f_statistic_and_p_value[1].round(3)
print(summary_statistic_table)


new_data = pd.DataFrame(data=[[1700, 2], [1800, 1]], columns=['SAT', 'Rand 1,2,3'])
new_data_scaled = scaler.transform(new_data)
print(reg.predict(new_data_scaled))


reg_simple = LinearRegression()
# print(x_features_scaled)
x_simple_matrix = x_features_scaled[:, 0].reshape(-1, 1)
# print(x_simple_matrix)
reg_simple.fit(x_simple_matrix, y)
print(reg_simple.predict(new_data_scaled[:, 0].reshape(-1, 1)))















