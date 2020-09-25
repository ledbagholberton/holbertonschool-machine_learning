#!/usr/bin/env python3
"""
The column Weighted_Price should be removed
missing values in High, Low, Open, and Close should be set to the previous
row’s Close value
missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.drop(labels="Weighted_Price", axis=1)
df = df["Volume_(BTC)"].fillna("0")
# YOUR CODE HERE

print(df.head())
print(df.tail())
