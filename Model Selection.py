import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    'Matches': [8, 8, 10, 8, 8, 8, 8],
    'Win': [6, 7, 10, 4, 7, 6, 7],
    'Drawn': [2, 1, 0, 2, 0, 1, 0],
    'Lost': [0, 0, 0, 2, 1, 0, 1],
    'Goals': [22, 29, 36, 16, 25, 22, 30],
    'Team Name': ['England', 'France', 'Portugal', 'Italy', 'Spain', 'Belgium', 'Germany'],
    'Goals conceded': [4, 3, 2, 9, 5, 4, 7],
    'PossessionAccuracy': ['62.63%', '60.13%', '63.10%', '58.80%', '67.30%', '58.13%', '60%'],
    'Balls Recovered': [273, 297, 422, 321, 276, 258, 299],
    'Clean Sheets': [4, 6, 9, 3, 4, 5, 5],
    'Saves': [5, 15, 16, 15, 11, 16, 14],
    'Yellow cards': [14, 12, 11, 14, 16, 13, 12],
    'Red Cards': [1, 0, 0, 0, 0, 1, 0],
    'PassingAccuracy': ['80%', '90%', '89%', '88%', '90%', '86%', '89%']
}

# Convert data to DataFrame
df = pd.DataFrame(data)

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

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42, probability=True),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    
    if y_pred_proba is not None:
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")
    print("\n")

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best ROC-AUC Score: {grid_search.best_score_:.2f}")

# Evaluate the best model on test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("Best Model Performance on Test Data:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")

# Feature importance visualization for Random Forest
importances = best_model.feature_importances_
features = X.columns

feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

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
