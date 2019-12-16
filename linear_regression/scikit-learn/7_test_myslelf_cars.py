import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import warnings
warnings.filterwarnings('ignore')


pd.set_option('max_columns', 20)
current_dir = os.getcwd()
# print(current_dir)
previous_dir = os.path.dirname(os.path.dirname(current_dir))   # os.path.dirname() method in Python is used to get the directory name from the specified path
# print(previous_dir)
df = pd.read_csv(os.path.join(previous_dir,
                              r'csv_files\1.04. Real-life example.csv'))

# print(df.head(2))
# print(df.describe(include='all'))

df = df.drop(['Model'], axis=1)
# print(df.describe(include='all'))
# print(df.isnull().sum())

df = df.dropna(axis=0)
# print(df.describe(include='all'))
# print(df.describe())

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 6))
# sns.distplot(df['Price'], ax=ax1)
# sns.distplot(df['Mileage'], ax=ax2)
# sns.distplot(df['EngineV'], ax=ax3)
# sns.distplot(df['Year'], ax=ax4)
# plt.show()

quantile_high_Price = df['Price'].quantile(0.99)
quantile_low_Price = df['Price'].quantile(0.05)
# print(quantile_low_Price, quantile_high_Price)
df = df[(df['Price'] > quantile_low_Price) & (df['Price'] < quantile_high_Price)]
# sns.distplot(df['Price'])
# plt.show()

quantile_high_Mileage = df['Mileage'].quantile(0.99)
quantile_low_Mileage = df['Mileage'].quantile(0.05)
# print(quantile_low_Mileage, quantile_high_Mileage)
# df = df[df['Mileage'] < quantile_high_Mileage]
df = df[(df['Mileage'] > quantile_low_Mileage) & (df['Mileage'] < quantile_high_Mileage)]
# sns.distplot(df['Mileage'])
# plt.show()

df = df[df['EngineV'] < 6.5]
# sns.distplot(df['EngineV'])
# plt.show()

quantile_high_Year = df['Year'].quantile(0.99)
quantile_low_Year = df['Year'].quantile(0.01)
# print(quantile_low_Year, quantile_high_Year)
df = df[(df['Year'] > quantile_low_Year) & (df['Year'] < quantile_high_Year)]
# sns.distplot(df['Year'])
# plt.show()

# print(df.describe())

df = df.reset_index(drop=True)
# print(df.describe())

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 6))
# sns.distplot(df['Price'], ax=ax1)
# sns.distplot(df['Mileage'], ax=ax2)
# sns.distplot(df['EngineV'], ax=ax3)
# sns.distplot(df['Year'], ax=ax4)
# plt.show()

log_price = np.log(df['Price'])
df['log_price'] = log_price
# print(df.head(3))
df.rename({'log_price': 'Log Price'}, axis=1, inplace=True)
# print(df.columns)
df = df.drop(['Price'], axis=1)


'''Multicolliniarity'''
from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = df[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF (Variance Inflatin Factor)'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
# print(vif)
df.drop(['Year'], axis=1, inplace=True)


''' Create Dummies Variables'''
df_with_dummies = pd.get_dummies(df, drop_first=True)
# print(df_with_dummies.head())


''' Rearrange a bit '''
columns = list(df.columns)
columns.remove('Log Price')
columns.insert(0, 'Log Price')
# print(columns)
df = df[columns]
# print(df.head(2))


''' Scale the data'''
from sklearn.preprocessing import StandardScaler
predicted_value = df_with_dummies['Log Price']
features_input = df_with_dummies.drop(['Log Price'], axis=1)

scaler = StandardScaler()
features_input_scaled = scaler.fit_transform(features_input)



''' Train Test Split'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_input_scaled, predicted_value, test_size=0.2, random_state=365)


''' Create the regression '''
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)
# print('Linear regreesion score ==>', regression.score(X_test, y_test))

yhat = regression.predict(X_train)
print('Linear regreesion test score ==>', regression.score(X_train, y_train),
      '\nLinear regreesion train score ==>', regression.score(X_test, y_test))

plt.scatter(y_train, yhat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_test)', size=18)
plt.show()

# sns.distplot(y_train - yhat)
# plt.title('Residual PDF', size=18)
# plt.show()

























































