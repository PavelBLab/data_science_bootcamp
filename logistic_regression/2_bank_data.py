import pandas as pd
pd.set_option('max_columns', 999)
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('../csv_files/Bank-data.csv')
# print(df.head())
df = df.drop(['Unnamed: 0'], axis=1)
print(df.head())

df['y'] = df['y'].map({'yes': 1, 'no': 0})
print(df.describe())


y = df['y']
x1 = df['duration']

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
print(reg_log.fit().summary())


# sns.regplot(x1, y, color='C0')
plt.scatter(x1, y, c='C0')
plt.xlabel('Duration')
plt.ylabel('Subscription')
# plt.show()

# the odds of duration are the exponential of the log odds from the summary table
import numpy as np
print(np.exp(0.0051))

''' Declare the independent variable(s) '''
X1 = df.drop(['y', 'may'], axis=1)
print(X1.columns)

X = sm.add_constant(X1)
reg_logit = sm.Logit(y, X)
result_logit = reg_logit.fit()
print(result_logit.summary())


def confusion_matrix(data, actual_values, model):
    # Confusion matrix

    # Parameters
    # ----------
    # data: data frame or array
    # data is a data frame formatted in the same way as your input data (without the actual values)
    # e.g. const, var1, var2, etc. Order is very important!
    # actual_values: data frame or array
    # These are the actual values from the test_data
    # In the case of a logistic regression, it should be a single column with 0s and 1s

    # model: a LogitResults object
    # this is the variable where you have the fitted model
    # e.g. results_log in this course
    # ----------

    # Predict the values using the Logit model
    pred_values = model.predict(data)
    # Specify the bins
    bins = np.array([0, 0.5, 1])
    # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
    # if they are between 0.5 and 1, they will be considered 1
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    # Calculate the accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    # Return the confusion matrix and
    return cm, accuracy

print(confusion_matrix(X, y, result_logit))














