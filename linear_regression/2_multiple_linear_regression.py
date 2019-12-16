import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import scipy as sp
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', 10)
df = pd.read_csv('../csv_files/1.02. Multiple linear regression.csv')
# print(df.head())
# print(df.describe())
print(df.columns)

y = df.loc[:, 'GPA']
x1 = df.loc[:, ['SAT', 'Rand 1,2,3']]
# print(y[:2])
# print(x1[:2])

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit().summary()
print(results)