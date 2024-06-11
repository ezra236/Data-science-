import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt 


path="team_statistics.csv"
df = pd.read_csv(path)


for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column Name:{column}\n")
        print(f"Data Types:{df[column].dtype}")
        print(f"Descriptive Statistics: {df[column].describe()}\n")
    else:
        if 'PossessionAccuracy' in df.columns:
           df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float') / 100.0
        if 'PassingAccuracy' in df.columns:
           df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float') / 100.0

print(df)          
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
