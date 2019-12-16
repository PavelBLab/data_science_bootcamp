import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
import scipy as sp
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 10) # Show all columns when looking at dataframe
pd.set_option('display.max_rows', 10) # Show all columns when looking at dataframe

df = pd.read_csv('../csv_files/1.01. Simple linear regression.csv')
print(df.head())
print(df.describe())

y = df.loc[:, 'GPA']
x1 = df.loc[:, 'SAT']

print(sp.corrcoef(x1, y))
print(np.corrcoef(x1, y))


# sn.scatterplot(x1, y)
plt.scatter(x1, y)
yhat = 0.0017 * x1 + 0.275
f = 0.0017 * 1850
print('>>>>>>', f)
fig = plt.plot(x1, yhat, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20, horizontalalignment='center')
plt.ylabel('GPA', fontsize=20, horizontalalignment='center')
# plt.show()

x = sm.add_constant(x1)
# print(x1)
# print(x)

results = sm.OLS(y, x).fit()
print(results.summary())






