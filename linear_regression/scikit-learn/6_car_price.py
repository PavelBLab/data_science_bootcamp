import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import os.path as path

csv_folder = path.abspath(path.join(path.join(__file__, "../../.."), 'csv_files'))
# print(path.abspath(__file__))  os.path.abspath(__file__) method in Python is used to get the file path and 2nd argument ../../.. move 3 levels up
# print(csv_folder)

df = pd.read_csv(path.join(csv_folder, '1.04. Real-life example.csv'))

print(df.head())
print(df.columns)
print(df.describe(include='all'))

df = df.drop(['Model'], axis=1)
print(df.isnull().sum())

df_no_missing_values = df.dropna(axis=0)
print(df_no_missing_values.isnull().sum())
print(df_no_missing_values.describe(include='all'))

# sns.distplot(df_no_missing_values['Price'])
# plt.show()

quantile = df_no_missing_values['Price'].quantile(0.99)
# print(quantile)
data_1 = df_no_missing_values[df_no_missing_values['Price'] < quantile]
# print(data_1.describe(include='all'))
# sns.distplot(data_1['Price'])
# plt.show()


quantile = df_no_missing_values['Mileage'].quantile(0.99)
# print(quantile)
data_2 = data_1[data_1['Mileage'] < quantile]
# print(data_2.describe(include='all'))
# sns.distplot(data_2['Mileage'])
# plt.show()

data_3 = data_2[data_2['EngineV'] < 6.5]
# sns.distplot(data_3['EngineV'])
# plt.show()

quantile = data_3['Year'].quantile(0.01)
# print(quantile)
data_4 = data_3[data_3['Year'] > quantile]
# print(data_4.describe(include='all'))
# sns.distplot(data_4['Year'])
# plt.show()

data_cleaned = data_4.reset_index(drop=True)
# print(data_cleaned.describe(include='all'))


# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 6))
# ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
# ax1.set_title('Price and Year')
# ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
# ax2.set_title('Price and EngineV')
# ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
# ax3.set_title('Price and Mileage')
# plt.show()


log_price = np.log(data_cleaned['Price'])
# print(log_price, log_price.shape)
data_cleaned['log_price'] = log_price
# print(data_cleaned)

data_cleaned.drop(['Price'], axis=1, inplace=True)
print(data_cleaned.head())

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 6))
# ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
# ax1.set_title('Log Price and Year')
# ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
# ax2.set_title('Log Price and EngineV')
# ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
# ax3.set_title('Log Price and Mileage')
# plt.show()


'''Multicolliniarity'''
from statsmodels.stats.outliers_influence import variance_inflation_factor
# print(data_cleaned.columns)
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
# print(variables.values)
# print(variables.shape)
vif['VIF (Variance Inflatin Factor)'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
# print(vif)

data_no_multicolliniarity = data_cleaned.drop('Year', axis=1)


''' Create Dummies Variables'''
data_with_dummies = pd.get_dummies(data_no_multicolliniarity, drop_first=True)
# print(data_with_dummies.head())

''' Rearrange a bit '''
cols = list(data_with_dummies.columns.values)
cols.remove('log_price')
cols.insert(0, 'log_price')
# print(col)
data_preprocessed = data_with_dummies[cols]
data_preprocessed.rename({'log_price': 'Log Price'}, axis=1, inplace=True)
# print(data_preprocessed.head())


''' Scale the data'''
predicted_value = data_preprocessed['Log Price']
features_input = data_preprocessed.drop(['Log Price'], axis=1)

scaler = StandardScaler()
scaler.fit(features_input)
features_input_scaled = scaler.transform(features_input)


''' Train Test Split'''
X_train, X_test, y_train, y_test = train_test_split(features_input_scaled, predicted_value, test_size=.2, random_state=365)

''' Create the regression '''
reg = LinearRegression()
reg.fit(X_train, y_train)

yhat = reg.predict(X_train)
print('Test score', reg.score(X_test, y_test))

plt.scatter(y_train, yhat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_test)', size=18)
plt.show()

# sns.distplot(y_train - yhat)
# plt.title('Residual PDF', size=18)
# plt.show()

print(list(data_preprocessed.drop(['Log Price', 'Mileage', 'EngineV'], axis=1).columns))
print(data_preprocessed['Brand_Mercedes-Benz'].unique())















































