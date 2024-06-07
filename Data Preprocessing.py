import pandas as pd
from IPython.display import display

# Load the CSV file into a DataFrame
input_csv_file = "team_statistics.csv"
df = pd.read_csv(input_csv_file)

# Data preprocessing
# Convert percentage strings to numerical values
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float') / 100
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float') / 100

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Display the DataFrame
display(df)

# Save the preprocessed DataFrame to a new CSV file
output_csv_file = "preprocessed_team_statistics.csv"
df.to_csv(output_csv_file, index=False)

print(f"Data successfully written to {output_csv_file}")
