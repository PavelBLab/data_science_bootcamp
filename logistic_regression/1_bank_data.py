import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('../csv_files/Example-bank-data.csv')
# print(df.head())

df['y'] = df['y'].map({'yes': 1, 'no': 0})
print(df.head())

y = df['y']
x = sm.add_constant(df['duration'])
# print(x)
reg_log = sm.Logit(y, x)
print(reg_log.fit().summary())


