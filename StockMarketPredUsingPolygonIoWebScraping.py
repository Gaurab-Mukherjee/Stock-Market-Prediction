import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from flask import *

# from flask_restful import Resource, Api
# import pickle
# from flask_cors import CORS

app = Flask(__name__)


@app.route('/close/', methods=['GET'])  # http://127.0.0.1:7777/close/?ticker=TSLA
def close_prediction():
    ticker = str(request.args.get('ticker'))
    print(ticker)
    api_key = "qbJ7w5ROSVLqBi3vYcGmY2Frqk29vrIn"
    # ticker = input("Enter the Company Name: ").upper()
    start = dt.date(2012, 1, 1)
    end = dt.date.today()
    data = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?apiKey={api_key}'
    data = requests.get(data).json()
    print(data)
    Close = pd.DataFrame(data['results'])
    # print(Close)
    # print(Close['c'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(Close['c'].values.reshape(-1, 1))

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
    model.fit(x_train, y_train, epochs=50, batch_size=32)
    loss, accuracy = model.evaluate(x_train, y_train)
    # print(f'Accuracy: {accuracy*100:.2f}%')
    # print(f'Loss: {loss*100:.2f}%')
    ''' Test The Model Accuracy on Existing Data '''

    # Load Test Data
    test_start = dt.date(2021, 1, 1)
    test_end = dt.date.today()

    test_data = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{test_start}/{test_end}?apiKey={api_key}'
    test_data = requests.get(test_data).json()
    test_close = pd.DataFrame(test_data['results'])
    actual_prices_close = test_close['c'].values

    total_dataset = pd.concat((Close['c'], test_close['c']), axis=0)

    model_inputs_close = 1

    model_inputs_close = total_dataset[len(total_dataset) - len(test_close) - prediction_days:].values
    model_inputs_close = model_inputs_close.reshape(-1, 1)
    model_inputs_close = scaler.transform(model_inputs_close)

    # Make Predictions on Test Data // Close
    x_test = []

    for x in range(prediction_days, len(model_inputs_close)):
        x_test.append(model_inputs_close[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    # print(f"Prediction1: {predicted_prices}")

    # Plot The Test Predictions
    plt.plot(actual_prices_close, color="black", label=f"Actual {ticker} Price")
    plt.plot(predicted_prices, color='green', label=f"Predicted {ticker} Price")
    plt.title(f"{ticker} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Share Price')
    plt.legend()
    plt.show()

    # Predict Next Day
    real_data = [model_inputs_close[len(model_inputs_close + 1) - prediction_days:len(model_inputs_close + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"PREDICTION OF THE NEXT DAY CLOSING PRICE : {prediction}")

    pred_set = {'Closing Price Prediction': f'{prediction}'}
    json_dump = json.dumps(pred_set)
    return json_dump


if __name__ == '__main__':
    app.debug = True
    app.run(port=7777)
