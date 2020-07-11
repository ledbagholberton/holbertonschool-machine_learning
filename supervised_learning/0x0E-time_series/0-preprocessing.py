"""
Data preprocessing:
Visually to predict data we can cut data from 2017
"""
import pandas as pd
import tensorflow as tf

def preprocessing(TRAIN_SPLIT):
	raw_data = pd.read_csv("./bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")
	df = raw_data.dropna()
	df.reset_index(inplace=True, drop=True)
	df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
	df["year"] = pd.DatetimeIndex(df["Timestamp"]).year
	df17 =  df[df["year"] >= 2017]
	df17.reset_index(inplace=True, drop=True)
	time_index = pd.DatetimeIndex(df17['Timestamp'])
	df_new=df17.set_index(time_index)
	df_new.drop('Timestamp',axis=1,inplace=True)
	df_new.drop('year',axis=1,inplace=True)
	df_new.drop('Open',axis=1,inplace=True)
	df_new.drop('Close',axis=1,inplace=True)
	df_new.drop('High',axis=1,inplace=True)
	df_new.drop('Low',axis=1,inplace=True)
	df_new['Volume_(BTC)'].resample('H').sum()
	df_new['Volume_(Currency)'].resample('H').sum()
	df_new['Weighted_Price'].resample('H').mean()
	features_considered=['Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
	features = df_new[features_considered]
	dataset = features.values
	data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
	data_std = dataset[:TRAIN_SPLIT].std(axis=0)
	dataset = (dataset-data_mean)/data_std
	return dataset
