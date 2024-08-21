import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# Load the data
path = "team_statistics.csv"
df = pd.read_csv(path)

# Preprocess percentage columns
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float') / 100.0
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float') / 100.0

# Store team names separately before dropping the column
team_names = df['Team Name']

# Drop the 'Team Name' column and target
df.drop('Team Name', axis=1, inplace=True)

# Define features and target
X = df.drop('Win', axis=1)
y = df['Win']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Re-train the model using all available data
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Generate all combinations of match pairs
match_pairs = list(combinations(team_names, 2))

# Function to predict the outcome of a match between two teams
def predict_match_outcome(team_a_idx, team_b_idx):
    # Extract the feature data for both teams
    team_a_features = X_scaled[team_a_idx].reshape(1, -1)
    team_b_features = X_scaled[team_b_idx].reshape(1, -1)
    
    # Predict the probabilities for both teams
    prob_a_win = model.predict_proba(team_a_features)[:, 1][0]
    prob_b_win = model.predict_proba(team_b_features)[:, 1][0]
    
    # Compare probabilities to determine the outcome
    if prob_a_win > prob_b_win:
        return f"{team_names.iloc[team_a_idx]} is more likely to win against {team_names.iloc[team_b_idx]}"
    elif prob_b_win > prob_a_win:
        return f"{team_names.iloc[team_b_idx]} is more likely to win against {team_names.iloc[team_a_idx]}"
    else:
        return f"The match between {team_names.iloc[team_a_idx]} and {team_names.iloc[team_b_idx]} is likely to be a draw"

# Predict and display the outcome for all match pairs
for team_a, team_b in match_pairs:
    team_a_idx = team_names[team_names == team_a].index[0]
    team_b_idx = team_names[team_names == team_b].index[0]
    result = predict_match_outcome(team_a_idx, team_b_idx)
    print(result)
