"""
Data preprocessing:
Visually to predict data we can cut data from 2017
"""
import pandas as pd

raw_data = pd.read_csv("./bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")
df = raw_data.dropna()
df.reset_index(inplace=True, drop=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

#print(df)
df.drop([0,1,2,3], inplace=True)
df.reset_index(inplace=True, drop=True)
df["closeDiff"] = df["Close"].diff()
df["year"] = pd.DatetimeIndex(df["Timestamp"]).year

df19 =  df[df["year"] >= 2019]
df19.reset_index(inplace=True, drop=True)
df19["year"] = pd.DatetimeIndex(df19["Timestamp"]).year
df19["month"] = pd.DatetimeIndex(df19["Timestamp"]).month
df19["day"] = pd.DatetimeIndex(df19["Timestamp"]).day
df19["hour"] = pd.DatetimeIndex(df19["Timestamp"]).hour
df19["minute"] = pd.DatetimeIndex(df19["Timestamp"]).minute
new_dict = {'new_year':[], 'new_month':[], 'new_day':[], 'new_hour':[],
            'open':[], 'close':[], 'USD_trans':[]}
new_open = 0
new_close = 0
for index, row in df19.iterrows():
    
    year = row['year']
    month = row['month']
    day = row['day']
    hour = row['hour']
    minute = row['minute']
    if hour != 23 and minute == 0:
        new_dict['new_year'].append(year)
        new_dict['new_month'].append(month)
        new_dict['new_day'].append(day)
        new_dict['new_hour'].append(hour)
        new_dict['open'].append(new_open)
        new_dict['close'].append(new_close)
        new_open = row['Open']
    elif minute == 58:
        new_close == row['Close']
print(new_dict)
#new_df = pd.DataFrame.from_dict(new_dict)
#new_df.to_csv('from19.csv', index=False)
#new_df.plot(x="new_hour",y="open")
#print(new_df)