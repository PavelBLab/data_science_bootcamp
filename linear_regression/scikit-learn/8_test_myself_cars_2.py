import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import os.path as path

csv_folder = path.abspath(path.join(path.join(__file__, "../../.."), 'csv_files'))
# print(path.abspath(__file__))  os.path.abspath(__file__) method in Python is used to get the file path and 2nd argument ../../.. move 3 levels up
# print(csv_folder)

pd.set_option('max_columns', 999)
df_original = pd.read_csv(path.join(csv_folder, '1.04. Real-life example.csv'))

# print(df_original.head())
# print(df_original.describe(include='all'))

df = df_original.copy()
# print(df.head())
# print(df.columns)

df = df.drop('Model', axis=1)
print(df.columns)

# print(df.isnull().sum())
df = df.dropna(axis=0)
# print(df.describe())

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 6))
# sns.distplot(df['Price'], ax=ax1)
# sns.distplot(df['Mileage'], ax=ax2)
# sns.distplot(df['EngineV'], ax=ax3)
# sns.distplot(df['Year'], ax=ax4)
# plt.show()

quantile_top_price = df['Price'].quantile(0.99)
quantile_bottom_price = df['Price'].quantile(0.01)
df = df[(df['Price'] > quantile_bottom_price) & (df['Price'] < quantile_top_price)]

quantile_top_Mileage = df['Mileage'].quantile(0.99)
quantile_bottom_Mileage = df['Mileage'].quantile(0.01)
df = df[(df['Mileage'] > quantile_bottom_Mileage) & (df['Mileage'] < quantile_top_Mileage)]

df = df[df['EngineV'] < 6.5]

quantile_top_Year = df['Year'].quantile(0.99)
quantile_bottom_Year = df['Year'].quantile(0.01)
df = df[(df['Year'] > quantile_bottom_Year) & (df['Year'] < quantile_top_Year)]

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 6))
# sns.distplot(df['Price'], ax=ax1)
# sns.distplot(df['Mileage'], ax=ax2)
# sns.distplot(df['EngineV'], ax=ax3)
# sns.distplot(df['Year'], ax=ax4)
# plt.show()

df = df.reset_index(drop=True)

log_price = np.log(df['Price'])
df['log_price'] = log_price
df.rename({'log_price': 'Log Price'}, axis=1, inplace=True)
df.drop('Price', axis=1, inplace=True)


''' Statistic '''
x_constant = sm.add_constant(df[['Mileage', 'EngineV', 'Year']])
reg = sm.OLS(df['Log Price'], x_constant).fit()
# print(reg.summary())


'''Multicolliniarity'''
from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = df[['Mileage', 'EngineV', 'Year']]
vif = pd.DataFrame()
# print(variables)
# print(variables.values)
vif['VIF (Variance Inflatin Factor)'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
# print(vif)
df.drop('Year', axis=1, inplace=True)

''' Create Dummies Variables'''
df = pd.get_dummies(df, drop_first=True)

''' Rearrange a bit '''
columns = list(df.columns)
# print(columns)
columns.remove('Log Price')
columns.insert(0, 'Log Price')
# print(columns)
df = df[columns]
# print(df.columns)

''' Scale the data'''
from sklearn.preprocessing import StandardScaler
# fig, (ax1, ax2) = plt.subplots(1, 2)
# sns.regplot(df['Mileage'], df['Log Price'], ax=ax1)
# sns.regplot(df['EngineV'], df['Log Price'], ax=ax2)
# plt.show()

predicted_value = df['Log Price']
# print(predicted_value.shape)
features_input = df.drop(['Log Price'], axis=1)
# print(features_input.shape)

scaler = StandardScaler()
scaler.fit(features_input)
# print(scaler.fit(features_input))
features_input_scaled = scaler.transform(features_input)
# print(df[['Mileage', 'EngineV']].head())
# print(features_input_scaled)

''' Train Test Split'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_input_scaled, predicted_value, test_size=0.2, random_state=365)


''' Create the regression '''
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
print(f'Linear regression train score = {linear_reg.score(X_train, y_train)}',
      f'\nLinear regression test score = {linear_reg.score(X_test, y_test)}',
      f'\nintercept = {linear_reg.intercept_}',
      f'\ncoef_ = {linear_reg.coef_}')


yhat = linear_reg.predict(X_train)
plt.scatter(y_train, yhat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_test)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
# plt.show()


''' Create a regression summary where we can compare them with one-another '''
reg_summary = pd.DataFrame(features_input.columns, columns=['Features'])
reg_summary['Weights'] = linear_reg.coef_
# print(reg_summary)

# Check the different categories in the 'Brand' variable
# print(df_original['Brand'].unique())



''' Testing '''
# Once we have trained and fine-tuned our model, we can proceed to testing it
# Testing is done on a dataset that the algorithm has never seen
# Luckily we have prepared such a dataset
# Our test inputs are 'x_test', while the outputs: 'y_test'
# We SHOULD NOT TRAIN THE MODEL ON THEM, we just feed them and find the predictions
# If the predictions are far off, we will know that our model overfitted
y_hat_test = linear_reg.predict(X_test)

# Create a scatter plot with the test targets and the test predictions
# You can include the argument 'alpha' which will introduce opacity to the graph
# plt.scatter(y_test, y_hat_test, alpha=0.2)
# plt.xlabel('Targets (y_test)',size=18)
# plt.ylabel('Predictions (y_hat_test)',size=18)
# plt.legend(['Test', 'Prediction'])
# plt.xlim(6,13)
# plt.ylim(6,13)
# plt.show()



''' Finally, let's manually check these predictions '''
# To obtain the actual prices, we take the exponential of the log_price
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
# print(df_pf.head())
df_pf['Target'] = np.exp(y_test)
# print(df_pf.head())

# After displaying y_test, we find what the issue is
# The old indexes are preserved (recall earlier in that code we made a note on that)
# The code was: data_cleaned = data_4.reset_index(drop=True)

# Therefore, to get a proper result, we must reset the index and drop the old indexing
y_test = y_test.reset_index(drop=True)
df_pf['Target'] = np.exp(y_test)
# print(df_pf.head())

''' OR '''
df_pf1 = pd.DataFrame({'Prediction': np.exp(y_hat_test), 'Target': np.exp(y_test)})
# print(df_pf1.head())

# Additionally, we can calculate the difference between the targets and the predictions
# Note that this is actually the residual (we already plotted the residuals)
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
# Finally, it makes sense to see how far off we are from the result percentage-wise
# Here, we take the absolute difference in %, so we can easily order the data frame
df_pf['Difference%'] = np.absolute(df_pf['Residual'] / df_pf['Target'] * 100)
# print(df_pf.head())
print(df_pf.describe())

# Sometimes it is useful to check these outputs manually
# To see all rows, we use the relevant pandas syntax
pd.options.display.max_rows = 999
# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot
pd.options.display.float_format = '{:.2f}%'.format
# Finally, we sort by difference in % and manually check the model
# df_pf.sort_values(by=['Difference%'])
df_pf.style.format({'Difference%': '{:.2%}'})
print(df_pf.head())










































































































