import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

# Load the trained model
model = load_model('power_prediction_model.h5')

# Load the scaler used to train the model
scaler = MinMaxScaler()


# Define a function to preprocess input data for the model
def prepare_input_data(data, scaler):
    # Scale the data using the same scaler used to train the model
    scaled_data = scaler.transform(data)
    # Reshape the data to match the input shape of the model
    # X = np.reshape(scaled_data, (1, 24, 9))
    return scaled_data


# Example input data for the forecast
forecast_data = np.array([[2023, 2, 17, 0, 3, 0, 50000, 25, 300]])

# Prepare the input data for the model
X_forecast = prepare_input_data(forecast_data, scaler)

# Generate predictions using the loaded model
y_pred = model.predict(X_forecast)

# Create a new DataFrame with the predicted values and any other relevant information
forecast_df = pd.DataFrame({'datetime': pd.date_range(start='2023-02-17', periods=24, freq='H'),
                            'power': y_pred.reshape(-1)})

# Save the DataFrame to a .xlsx file
forecast_df.to_excel('power_forecast.xlsx', index=False)
