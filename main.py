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
# boosting gradient
features = ['year', 'month', 'day', 'hour', 'week_day', 'holiday_type', 'bitcoin', 'temperature', 'power']
df = df[features]
# extract required columns

# scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
print(scaled_data, 'scaled data')
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]
n_steps = 24  # 24 hours


def create_sequences(data, steps):
    X, y = [], []
    for i in range(steps, len(data)):
        X.append(data[i - steps:i])
        y.append([data[i]])  # return target variable as a 2D array
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_data, n_steps)
X_test, y_test = create_sequences(test_data, n_steps)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, len(features))))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[checkpoint])
