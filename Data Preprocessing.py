import pandas as pd


path="team_statistics.csv"
df = pd.read_csv(path)

i=10

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

          
