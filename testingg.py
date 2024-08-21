import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data again to include the 'Team Name' column
path = "team_statistics.csv"
df = pd.read_csv(path)

# Preprocess percentage columns
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float') / 100.0
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float') / 100.0

# Define features and target
features = ['Red Cards', 'PassingAccuracy', 'Balls Recovered', 'Goals', 'PossessionAccuracy']
X = df[features]
y = df['Win']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Create all possible pairs of teams
teams = df['Team Name'].unique()
matchups = list(combinations(teams, 2))

# Predict the outcome for each matchup
predictions = []

for team1, team2 in matchups:
    # Extract the features for both teams
    team1_features = df[df['Team Name'] == team1][features].values
    team2_features = df[df['Team Name'] == team2][features].values

    # Predict the probability of winning for each team
    team1_prob = model.predict_proba(scaler.transform(team1_features))[:, 1][0]
    team2_prob = model.predict_proba(scaler.transform(team2_features))[:, 1][0]

    # Determine the predicted winner
    if team1_prob > team2_prob:
        winner = team1
        win_prob = team1_prob
    else:
        winner = team2
        win_prob = team2_prob
    
    # Store the prediction
    predictions.append({
        'Team 1': team1,
        'Team 2': team2,
        'Predicted Winner': winner,
        'Win Probability': win_prob
    })

# Convert predictions to a DataFrame for better visualization
predictions_df = pd.DataFrame(predictions)

# Display the predictions
print(predictions_df)
