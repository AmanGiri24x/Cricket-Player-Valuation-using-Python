import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define feature columns based on dataset columns
FEATURE_COLUMNS = [
    'Year', 'Matches', 'Runs', 'Batting_Strike_Rate', 'Wickets_Taken',
]

TARGET_COLUMN = 'Valuation'  # This is the target you're predicting

# Step 1: Data Loading and Preprocessing
def load_and_prepare_data(file_path):
    logger.info("Loading dataset...")
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully from {file_path}.")
        
        # Handling missing values
        data = data.fillna(0)
        
        # Ensure required columns exist
        missing_columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

# Step 2: Model Training
def train_and_save_model(data):
    logger.info("Starting model training...")
    try:
        # Define features (X) and target (y)
        X = data[FEATURE_COLUMNS]
        y = data[TARGET_COLUMN]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully with MAE: {mae:.4f} and R²: {r2:.4f}")
        
        # Save the scaler and model
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(model, 'models/best_player_valuation_model.pkl')
        logger.info("Model and scaler saved successfully.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

# Step 3: Load Model and Scaler
def load_model_and_scaler():
    logger.info("Loading model and scaler...")
    try:
        model = joblib.load('models/best_player_valuation_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        logger.info("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model/scaler: {e}")
        raise

# Step 4: Streamlit Application for Player Valuation Prediction
def run_app():
    st.title("Cricket Player Valuation Predictor")
    st.write("Predict the valuation of a cricket player based on their performance metrics.")
    
    # Load dataset
    data = load_and_prepare_data("realistic_merged_data.csv")  # Ensure correct path to dataset file
    
    # Player Selection from dropdown
    player_name = st.selectbox("Select Player", data['Player'].unique())
    
    if player_name:
        player_data = data[data['Player'] == player_name]
        st.subheader(f"Player Stats for {player_name}:")
        st.table(player_data[FEATURE_COLUMNS + [TARGET_COLUMN]])  # Display selected player stats

        # Load the model and scaler
        try:
            model, scaler = load_model_and_scaler()
        except:
            st.error("Model or scaler not found. Please ensure training is complete.")
            return

        # Predict valuation using the player's stats
        player_features = player_data[FEATURE_COLUMNS].values
        player_features_scaled = scaler.transform(player_features)
        predicted_valuation = model.predict(player_features_scaled)

        # Performance-based Valuation Adjustment
        valuation = predicted_valuation[0]

        # Normalize performance data against dataset max values
        max_runs = data['Runs'].max()
        max_wickets = data['Wickets_Taken'].max()
        max_matches = data['Matches'].max()

        normalized_runs = player_data['Runs'].values[0] / max_runs
        normalized_wickets = player_data['Wickets_Taken'].values[0] / max_wickets
        normalized_matches = player_data['Matches'].values[0] / max_matches

        # Role-based weighting factors (tune these as needed)
        runs_weight = 0.5
        wickets_weight = 0.3
        matches_weight = 0.2

        # Adjust valuation based on normalized performance (to prevent runaway values)
        valuation = (
            (normalized_runs * runs_weight) +
            (normalized_wickets * wickets_weight) +
            (normalized_matches * matches_weight)
        ) * valuation

        # Apply IPL base price constraint (₹30 lakh as starting price)
        base_price = 0.3  # ₹30 lakh = ₹0.3 Crore
        
        # Adjust valuation based on number of matches (experience)
        experience_factor = player_data['Matches'].values[0] / 100  # Adjust the divisor as per dataset
        valuation += experience_factor * 0.1  # You can modify the weight of the experience factor
        
        # Prevent valuation from going too high (controlled scaling)
        valuation = min(valuation, 30)  # Set a hard max of ₹30 Crore
        valuation = max(valuation, base_price)  # Ensure base price of ₹30 lakh (₹0.3 Crore) is met
        
        # Prevent runaway valuation for average players (like Lalit Yadav)
        if valuation > 15:
            # Apply a cap on valuation based on player performance
            valuation = 15 + (valuation - 15) * 0.3  # Dampens extreme valuations for players with poor stats
        
        # Display valuation
        valuation_display = f"₹{valuation:.2f} Crores"
        st.subheader(f"Predicted Valuation: {valuation_display}")

# Main entry point
if __name__ == "__main__":
    try:
        # Uncomment the next line to train the model (only required once)
       train_and_save_model(load_and_prepare_data("realistic_merged_data.csv"))
        
       run_app()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An unexpected error occurred: {e}")
