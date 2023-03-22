import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the trained model from a file
model = load_model('power_prediction_model.h5')

# Load the input data
data = pd.read_csv('final_data.csv', parse_dates=['datetime'], index_col=['datetime'])
data['missing_feature'] = 0
# Extract the input features
X = data[['year', 'month', 'day', 'hour', 'week_day', 'holiday_type', 'bitcoin', 'temperature', 'power']]
n_steps = 24

# Scale the input data to the range [0, 1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Initialize an array to hold the predicted power values
start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2023-02-01')
n_hours = int((end_date - start_date).total_seconds() / 3600)
predicted_power = np.zeros((n_hours, 1))

# Make hourly predictions for January 2023
for i in range(n_hours):
    # Get the input data for the current hour
    hour_data = X[i:i+n_steps, :]
    hour_data = np.reshape(hour_data, (1, n_steps, 5))

    # Make a prediction using the model
    hour_power = model.predict(hour_data)

    # Invert the scaling to get the actual power value
    # hour_power = scaler.inverse_transform([[0, 0, 0, hour_power]])[0][3]
    # hour_power = scaler.inverse_transform([0, 0, 0, 0, hour_power])
    print(hour_power)
    # Add the predicted power to the array
#     predicted_power[i, 0] = hour_power
#
# # Save the predicted power values to a file
# dates = pd.date_range(start=start_date, end=end_date, freq='H')
# df = pd.DataFrame(predicted_power, index=dates, columns=['power'])
# df.to_excel('january_power_forecast.xlsx')
