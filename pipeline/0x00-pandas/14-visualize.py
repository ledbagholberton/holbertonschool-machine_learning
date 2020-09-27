#!/usr/bin/env python3
"""
Plot the data from 2017 and beyond at daily intervals
The column Weighted_Price should be removed
Rename the column Timestamp to Date
Convert the timestamp values to date values
Index the data frame on Date
Missing values in High, Low, Open, and Close should be set to the previous
row’s Close value
Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
"""
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.drop(["Weighted_Price"], axis=1)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.rename(columns={'Timestamp': 'Date'})
df[['High', 'Low', 'Open', 'Close']] = df[['High', 'Low', 'Open', 'Close']].fillna(method='ffill')
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(value = 0)
df=df.set_index('Date')
df1=df[(df.index>'2017-01-01')]
df1=df1.resample('D').mean()
df1.plot()
plt.show()
