import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('../csv_files/1.03. Dummies.csv')
# print(df.drop(['Attendance'], axis=1))
f = lambda x: 1 if x == 'Yes' else 0
# df['Attendance_dummy'] = df['Attendance'].apply(f)
df['Attendance_dummy'] = df['Attendance'].map({'Yes': 1, 'No': 0})

print(df)
# print(df.describe())
print(df.columns)

y = df.loc[:, 'GPA']
x1 = df.loc[:, ['SAT', 'Attendance_dummy']]

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit().summary()
print(results)

plt.scatter(df['SAT'], y, c=df['Attendance_dummy'], cmap='jet')
scatter = sns.regplot(df['SAT'], y, scatter=False)
print([scatter.get_children()[3]])
# print(plt.gca().patches)

yhat_yes = 0.8665 + 0.0014 * df['SAT']
yhat_no = 0.6439 + 0.0014 * df['SAT']
plt.plot(df['SAT'], yhat_yes, c='green')
plt.plot(df['SAT'], yhat_no, c='red')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()
