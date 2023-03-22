from scipy.stats import spearmanr
import pandas as pd
# Load the dataset into a pandas dataframe
df = pd.read_csv('final_data.csv')
print(df.iloc[:, 0])
# Calculate the Spearman correlation coefficient for each independent variable
# for col in range(5):
#     spearman_coef, p_value = spearmanr(df.iloc[:, col], df.iloc[:, 5])
#     print(f"Spearman correlation coefficient between column {col+1} and column 6: {spearman_coef}, p-value: {p_value}")
