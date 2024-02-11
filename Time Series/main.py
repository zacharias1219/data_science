import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

df = pd.read_csv('gold_monthly_csv.csv')
print(df.head())

print(df.shape)

print(f"Data is available from - {df.loc[:,'Date'][0]} to {df.loc[:,'Date'][len(df)-1]}")

date = pd.date_range(start='1/1/1950', end='8/1/2020', freq='M')
print(date)

df['month'] = date
df.drop('Date', axis=1, inplace=True)
df = df.set_index('month')
print(df.head())

df.plot(figsize=(20,8))
plt.title("Gold Price monthly since 1950 and onwards")

plt.xlabel("months")
plt.ylabel("price in USD")
plt.grid()

round(df.describe(),3)

_, ax = plt.subplots(figsize=(25, 8))
sns.boxplot(x=df.index.year,y=df.values[:,0], ax=ax)
