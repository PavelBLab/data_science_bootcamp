import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 10)
df = pd.read_csv('../csv_files/real_estate_price_size_year.csv')
print(df.head(2))

y = df.loc[:, 'price']
x1 = df.loc[:, 'size']

sns.regplot(x1, y)
# plt.show()

x1_constant = sm.add_constant(x1)
results1 = sm.OLS(y, x1_constant).fit().summary()
print(results1)


x2 = df.loc[:, ['size', 'year']]
x2_constant = sm.add_constant(x2)
# print(x2_constant)
results2 = sm.OLS(y, x2_constant).fit().summary()
print(results2)
