from sklearn.metrics import normalized_gini_score
import pandas as pd
# Load the dataset into a pandas dataframe
df = pd.read_csv('final_data.csv')

# Calculate the Gini index for each independent variable
for col in range(5):
    gini_index = normalized_gini_score(df.iloc[:, col], df.iloc[:, 5])
    print(f"Gini index between column {col+1} and column 6: {gini_index}")
