import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

# Define a list of models to evaluate
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Evaluate each model using stratified cross-validation with 3 splits
results = {}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    results[model_name] = cv_scores
    print(f"{model_name}: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train and evaluate the best model
best_model_name = max(results, key=lambda k: np.mean(results[k]))
best_model = models[best_model_name]
best_model.fit(X_scaled, y)

# Print the evaluation metrics
print(f"\nBest Model: {best_model_name}")
print("Training Accuracy:", np.mean(results[best_model_name]))

# Plot the feature importances for RandomForestClassifier
if isinstance(best_model, RandomForestClassifier):
    feature_importances = pd.DataFrame(best_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.importance, y=feature_importances.index)
    plt.title('Feature Importances')
    plt.show()
