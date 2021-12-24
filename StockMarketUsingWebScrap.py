#Importing the Libraries
import pandas as PD
import numpy as np
# %matplotlib inline
import matplotlib. pyplot as plt
import matplotlib
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import adam_v2
from keras.models import load_model
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model


# Get the Dataset
df = PD.read_csv("MSFT.csv", na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True)
df.head()

# Print the shape of Dataframe  and Check for Null Values
print("Dataframe Shape: ", df.shape)
print("Null Value Present: ", df.values.flatten())

# Plot the True Adj Close Value
df['Adj Close'].plot()

# Set Target Variable
output_var = PD.DataFrame(df['Adj Close'])
# Selecting the Features
features = ['Open', 'High', 'Low', 'Volume']

# Scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform = PD.DataFrame(columns=features, data=feature_transform, index=df.index)
feature_transform.head()

# Splitting to Training set and Test set
timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[
                                                            len(train_index): (len(train_index) + len(test_index))]
    y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (
                len(train_index) + len(test_index))].values.ravel()

    # Process the data for LSTM
    trainX = np.array(X_train)
    testX = np.array(X_test)
    X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Building the LSTM Model
    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)
    # plot_model(lstm, show_shapes=True, show_layer_names=True)

    # LSTM Prediction
    y_pred = lstm.predict(X_test)
    # y_pred = scaler.inverse_transform(y_pred)

    # Predicted vs True Adj Close Value – LSTM
    plt.plot(y_test, color='red', label='Actual Stock Price')
    plt.plot(y_pred, color='green', label='Predicted Stock Price')
    plt.title("Prediction by LSTM")
    plt.xlabel('TimeScale')
    plt.ylabel('Scaled USD')
    plt.legend()
    plt.show()
