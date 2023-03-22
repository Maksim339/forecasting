import pandas as pd

df = pd.read_csv('final_data.csv')

features = []

col1 = df['week_day']
col2 = df['power']

correlation_coefficient = col1.corr(col2)

print("The correlation coefficient between column1 and column2 is:", correlation_coefficient)

