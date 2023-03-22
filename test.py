import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# load the trained model
model = load_model('power_prediction_model.h5')

# load the data
df = pd.read_csv('final_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

df = df.drop(['datetime'], axis=1)
# extract required columns
features = ['year', 'month', 'day', 'hour', 'week_day', 'holiday_type', 'bitcoin', 'temperature', 'power']
df = df[features]
n_steps = 24
# scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# create a sequence of inputs for the next six months
# last_date = df['datetime'].max()
next_dates = pd.date_range('2023-02-17', periods=6*30*24+1, freq='H')[1:]
next_inputs = np.zeros((len(next_dates), n_steps, len(features)))
for i, date in enumerate(next_dates+1):
    if i < n_steps:
        next_inputs[i] = scaled_data[-n_steps+i:]
    else:
        next_inputs[i] = next_inputs[i-n_steps:i]
        next_inputs[i][-1,0] = date.year
        next_inputs[i][-1,1] = date.month
        next_inputs[i][-1,2] = date.day
        next_inputs[i][-1,3] = date.hour
        next_inputs[i][-1,4] = date.weekday()
        next_inputs[i][-1,5] = 0
        next_inputs[i][-1,6] = 0
        next_inputs[i][-1,7] = np.nan
        next_inputs[i][-1,8] = np.nan

# generate the predictions
next_predictions = model.predict(next_inputs)

# unscale the data
next_predictions = scaler.inverse_transform(np.hstack((next_inputs[:,:,-1], next_predictions)))
next_predictions = next_predictions[:,-1]

# create a dataframe with the predicted values and dates
next_dates = next_dates[:-1]
next_df = pd.DataFrame({'datetime': next_dates, 'power': next_predictions})

# save the dataframe to an Excel file
next_df.to_excel('power_forecast.xlsx', index=False)
