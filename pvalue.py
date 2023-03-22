import statsmodels.api as sm
import pandas as pd

# Load the dataset into a pandas dataframe
df = pd.read_csv('final_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df = df.drop(['datetime'], axis=1)

# Split the dataset into X (the independent variables) and y (the dependent variable)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Add a constant to the X dataframe (for the intercept term in the regression)
X = sm.add_constant(X)

# Perform the multiple linear regression analysis
model = sm.OLS(y, X).fit()

# Print the summary of the regression results
print(model.summary())
