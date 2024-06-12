import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


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
# Feature importance using Random Forest
X = df.drop('Win', axis=1)
y = df['Win']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a Random Forest to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True)
rf.fit(X_scaled, y)

# Get feature importances
importances = rf.feature_importances_
features = X.columns

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

# Plotting the feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()



# Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
