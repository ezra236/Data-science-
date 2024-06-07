import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data from CSV
csv_file = "team_statistics.csv"
df = pd.read_csv(csv_file)

# Check for missing values
print(df.isnull().sum())

# Data preprocessing
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float')
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float')

# Define features and target
X = df.drop(['Team Name', 'Win'], axis=1)
y = df['Win']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define logistic regression model
logistic_model = LogisticRegression()

# Train the logistic regression model using the training set
logistic_model.fit(X_train, y_train)

# Predict the target values for the testing set
y_pred = logistic_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

print(y_pred)
