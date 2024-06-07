import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Load the data from CSV
csv_file = "team_statistics.csv"
df = pd.read_csv(csv_file)

# Data preprocessing
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float')
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float')

# Define features and target
X = df.drop(['Team Name', 'Win'], axis=1)
y = df['Win']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define logistic regression model
logistic_model = LogisticRegression()

# Define stratified k-fold cross-validator with 2 splits
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(logistic_model, X_scaled, y, cv=skf, scoring='accuracy')

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
