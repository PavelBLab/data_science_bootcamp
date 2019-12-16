import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('../csv_files/real_estate_price_size_year_view.csv')
# print(df)

df['view'] = df['view'].apply(lambda x: 1 if x == 'Sea view' else 0)
# print(df)

y = df['price']
x1 = df[['size', 'year', 'view']]

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit().summary()
print(results)


plt.scatter(df['size'], y, c=df['view'], cmap='viridis')
# sns.regplot(df['size'], y, scatter_kws={'c': "df['view']", 'cmap':'viridis'})
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()


