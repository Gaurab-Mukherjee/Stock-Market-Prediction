import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

api_key = "qbJ7w5ROSVLqBi3vYcGmY2Frqk29vrIn"
ticker = input("Enter the Company Name: ").upper()
start = dt.date(2020, 1, 1)
end = dt.date.today()
data = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?apiKey={api_key}'
data = requests.get(data).json()
print(data)
High = pd.DataFrame(data['results'])
# print(Close)
print(High['h'])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(High['h'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build The Model for close
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=10)
loss, accuracy = model.evaluate(x_train, y_train)
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Loss: {loss*100:.2f}%')
''' Test The Model Accuracy on Existing Data '''

# Load Test Data
test_start = dt.date(2021, 1, 1)
test_end = dt.date.today()

test_data = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{test_start}/{test_end}?apiKey={api_key}'
test_data = requests.get(test_data).json()
test_high = pd.DataFrame(test_data['results'])
actual_prices_close = test_high['h'].values

total_dataset = pd.concat((High['h'], test_high['h']), axis=0)

model_inputs_high = 1

model_inputs_high = total_dataset[len(total_dataset) - len(test_high) - prediction_days:].values
model_inputs_high = model_inputs_high.reshape(-1, 1)
model_inputs_high = scaler.transform(model_inputs_high)

# Make Predictions on Test Data // Close
x_test = []

for x in range(prediction_days, len(model_inputs_high)):
    x_test.append(model_inputs_high[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
print(f"Prediction1: {predicted_prices}")

# Plot The Test Predictions
plt.plot(actual_prices_close, color="black", label=f"Actual {ticker} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {ticker} Price")
plt.title(f"{ticker} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{ticker} Share Price')
plt.legend()
plt.show()

# Predict Next Day
real_data = [model_inputs_high[len(model_inputs_high+1) - prediction_days:len(model_inputs_high+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"PREDICTION OF THE NEXT DAY High PRICE : {prediction}")

