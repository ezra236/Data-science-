import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Creating the DataFrame
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
df = pd.DataFrame(data)

# Convert percentage strings to float
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype(float)
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype(float)

# Dropping 'Team Name' as it's categorical and non-numeric
df.drop('Team Name', axis=1, inplace=True)

# Correlation analysis for Numerical Input, Numerical Output (with 'Win')
correlations = {}
for col in df.columns:
    if col != 'Win':
        pearson_corr, _ = pearsonr(df[col], df['Win'])
        spearman_corr, _ = spearmanr(df[col], df['Win'])
        correlations[col] = {'Pearson': pearson_corr, 'Spearman': spearman_corr}

correlations_df = pd.DataFrame(correlations).T
print(correlations_df)

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
