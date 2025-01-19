import pandas as pd

# Load the CSV file
df = pd.read_csv('PatronusAIfinancebench.csv')

# Print the DataFrame
print(df.to_string(index=False))
