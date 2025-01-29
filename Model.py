import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load datasets
cricket_data = pd.read_csv("cricket_data.csv")
ipl_stats = pd.read_csv("IPL Player Stat.csv")

# Step 2: Rename columns for consistency
cricket_data.rename(columns={"Player Name": "Player"}, inplace=True)
ipl_stats.rename(columns={"Player Name": "Player"}, inplace=True)

# Step 3: Merge datasets
merged_data = pd.merge(cricket_data, ipl_stats, on='Player', how='inner')

# Step 4: Feature Engineering
merged_data['batting_impact'] = merged_data['runs'] * merged_data['batting_strike_rate'] / 100
merged_data['bowling_impact'] = merged_data['wickets'] / (merged_data['bowling_avg'] + 1)
merged_data['fielding_impact'] = merged_data['catches'] + merged_data['stumpings'] * 2
merged_data['experience'] = merged_data['matches']

# Ensure no leakage
if 'Valuation' in merged_data.columns:
    del merged_data['Valuation']  # Remove if exists

# Create target variable for valuation
merged_data['Valuation'] = (
    merged_data['batting_impact'] * 0.4 +
    merged_data['bowling_impact'] * 0.4 +
    merged_data['fielding_impact'] * 0.1 +
    merged_data['experience'] * 0.1
)

# Remove players with unrealistic stats (e.g., zero impact)
merged_data = merged_data[
    (merged_data['batting_impact'] > 0) |
    (merged_data['bowling_impact'] > 0) |
    (merged_data['fielding_impact'] > 0)
]

# Step 5: Define features and target
features = ['matches', 'batting_impact', 'bowling_impact', 'fielding_impact', 'experience']
target = 'Valuation'

X = merged_data[features]
y = merged_data[target]

# Step 6: Preprocess features (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Hyperparameter Tuning (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5, 10]
}

model = GradientBoostingRegressor(random_state=42)

# GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

# Best parameters from grid search
best_model = grid_search.best_estimator_

# Step 9: Evaluate the model on test data
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Model Hyperparameters: {grid_search.best_params_}")
print(f"Test Set Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Step 10: Save the trained model and scaler
joblib.dump(best_model, "realistic_player_valuation_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save the merged dataset
merged_data.to_csv("realistic_merged_data.csv", index=False)
print("Model and dataset saved successfully.")
