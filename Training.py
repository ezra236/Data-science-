import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load the data
path = "team_statistics.csv"
df = pd.read_csv(path)

# Preprocess percentage columns
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float') / 100.0
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float') / 100.0

# Drop the categorical 'Team Name' column
df.drop('Team Name', axis=1, inplace=True)

# Define features and target
X = df.drop('Win', axis=1)
y = df['Win']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability predictions for ROC-AUC score

# Evaluate the model's performance
print("Logistic Regression Model Performance on Test Data:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")
