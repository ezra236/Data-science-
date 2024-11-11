import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the data
path = "output_file.csv"
df = pd.read_csv(path)

# Preprocess percentage columns
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float') / 100.0
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float') / 100.0

# Store team names and goals separately before dropping columns
team_names = df['Team Name']
avg_goals_scored = df['Goals'] / 38  # Average goals scored per game
avg_goals_conceded = df['Goals'] / 38  # Average goals conceded per game

# Drop the 'Team Name' column and target
df.drop(['Team Name', 'Goals', 'Goals'], axis=1, inplace=True)

# Define features and target
X = df.drop('Win', axis=1)
y = df['Win']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model using all available data
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Generate all combinations of match pairs
match_pairs = list(combinations(team_names, 2))

# Function to predict a realistic match score between two teams
def predict_match_score(team_a_idx, team_b_idx):
    # Calculate expected goals based on average goals scored and conceded
    team_a_avg_goals = (avg_goals_scored[team_a_idx] + avg_goals_conceded[team_b_idx]) / 2
    team_b_avg_goals = (avg_goals_scored[team_b_idx] + avg_goals_conceded[team_a_idx]) / 2
    
    # Generate scores based on expected goals with randomness to simulate realistic outcomes
    team_a_score = np.random.poisson(team_a_avg_goals)
    team_b_score = np.random.poisson(team_b_avg_goals)
    
    return f"{team_names.iloc[team_a_idx]} {team_a_score} - {team_b_score} {team_names.iloc[team_b_idx]}"

# Predict and display the score for all match pairs
for team_a, team_b in match_pairs:
    team_a_idx = team_names[team_names == team_a].index[0]
    team_b_idx = team_names[team_names == team_b].index[0]
    result = predict_match_score(team_a_idx, team_b_idx)
    print(result)
