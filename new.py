import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

df = pd.read_csv('final_data.csv')

df['datetime'] = pd.to_datetime(df['datetime'])

df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

df = df.drop(['datetime'], axis=1)

# select relevant features
features = ['year', 'month', 'day', 'hour', 'week_day', 'holiday_type', 'bitcoin', 'temperature', 'power']
df = df[features]

# scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

n_steps = 24  # 24 hours

# create sequences of data for input to LSTM model
def create_sequences(data, steps):
    X, y = [], []
    for i in range(steps, len(data)):
        X.append(data[i - steps:i])
        y.append([data[i]])  # return target variable as a 2D array
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, n_steps)
X_test, y_test = create_sequences(test_data, n_steps)

# define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, len(features))))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# train the LSTM model
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[checkpoint])

# load the best saved model
best_model = load_model('best_model.h5')

# predict on the remaining 20% of the data
X_forecast, y_forecast = create_sequences(test_data, n_steps)
y_forecast_pred = best_model.predict(X_forecast)

# scale the predicted values back to their original range
y_forecast_pred_scaled = scaler.inverse_transform(y_forecast_pred)

# create a dataframe with the forecasted power values
df_forecast = pd.DataFrame(y_forecast_pred_scaled, columns=['power'])

# save the forecast to an excel file
df_forecast.to_excel('forecasted_power.xlsx', index=False)

