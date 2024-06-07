import pandas as pd
from IPython.display import display

# Data provided
data = {
    'Matches': [8, 8, 10, 8, 8, 8, 8],
    'Win': [6, 7, 10, 4, 7, 6, 7],
    'Drawn': [2, 1, 0, 2, 0, 1, 0],
    'Lost': [0, 0, 0, 2, 1, 0, 1],
    'Goals': [22, 29, 36, 16, 25, 22, 30],
    'Team Name': ['England', 'France', 'Portugal', 'Italy', 'Spain', 'Belgium', 'Germany'],
    'Goals conceded': [4, 3, 2, 9, 5, 4, 7],
    'PossessionAccuracy': ['62.63%', '60.13%', '63.10%', '58.80%', '67.30%', '58.13%', '60%'],
    'Balls Recovered': [273, 297, 422, 321, 276, 258, 299],
    'Clean Sheets': [4, 6, 9, 3, 4, 5, 5],
    'Saves': [5, 15, 16, 15, 11, 16, 14],
    'Yellow cards': [14, 12, 11, 14, 16, 13, 12],
    'Red Cards': [1, 0, 0, 0, 0, 1, 0],
    'PassingAccuracy': ['80%', '90%', '89%', '88%', '90%', '86%', '89%']
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file = "team_statistics.csv"
df.to_csv(csv_file, index=False)

# Display the DataFrame
display(df)

print(f"Data successfully written to {csv_file}")
