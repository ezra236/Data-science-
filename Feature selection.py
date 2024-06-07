import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the path to your CSV file
csv_file = "team_statistics.csv"


df = pd.read_csv(csv_file)


display(df)

# Convert percentage strings to float
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float')
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float')

# Check correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Feature importance using RandomForest
# Assuming the target variable is 'Win'
X = df.drop(['Team Name', 'Win'], axis=1)
y = df['Win']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances.importance, y=feature_importances.index)
plt.title('Feature Importances')
plt.show()

print(f"Data successfully written to {csv_file}")
