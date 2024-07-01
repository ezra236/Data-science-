import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preparation
path="team_statistics.csv"
df = pd.read_csv(path)

# Convert percentage strings to float
df['PossessionAccuracy'] = df['PossessionAccuracy'].str.rstrip('%').astype('float') / 100.0
df['PassingAccuracy'] = df['PassingAccuracy'].str.rstrip('%').astype('float') / 100.0

# Dropping 'Team Name' as it's categorical and non-numeric
df.drop('Team Name', axis=1, inplace=True)

# Selecting important features
features = ['Red Cards', 'PassingAccuracy', 'Balls Recovered', 'Goals', 'PossessionAccuracy']
X = df[features]
y = df['Win']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Dictionary
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Cross-validation and evaluation
results = {}
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    results[name] = {
        'CV Accuracy': np.mean(cv_scores),
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'ROC-AUC': roc_auc_score(y_test, y_prob, multi_class='ovr') if y_prob is not None and y_prob.shape[1] == len(np.unique(y_test)) else 'N/A'
    }

# Display results
results_df = pd.DataFrame(results).T
print(results_df)

# Hyperparameter Tuning Example with Grid Search for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=skf, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model after Grid Search
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
y_prob_best_rf = best_rf.predict_proba(X_test)
best_rf_results = {
    'Best RF Test Accuracy': accuracy_score(y_test, y_pred_best_rf),
    'Best RF Precision': precision_score(y_test, y_pred_best_rf, average='weighted'),
    'Best RF Recall': recall_score(y_test, y_pred_best_rf, average='weighted'),
    'Best RF F1 Score': f1_score(y_test, y_pred_best_rf, average='weighted'),
    'Best RF ROC-AUC': roc_auc_score(y_test, y_prob_best_rf, multi_class='ovr') if y_prob_best_rf.shape[1] == len(np.unique(y_test)) else 'N/A'
}

print("Best Random Forest Model Results:", best_rf_results)
