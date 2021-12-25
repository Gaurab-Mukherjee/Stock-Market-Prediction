import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


api_key = "qbJ7w5ROSVLqBi3vYcGmY2Frqk29vrIn"
ticker = input("Enter the Company Name: ").upper()
start = dt.date(2012, 1, 1)
end = dt.date.today()
api_url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?apiKey={api_key}'
print(api_url)
data = requests.get(api_url).json()
print(data)

df = pd.DataFrame(data['results'])
print(df)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['c'].values.reshape(-1, 1))

