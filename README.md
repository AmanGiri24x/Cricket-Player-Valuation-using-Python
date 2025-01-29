# Cricket-Player-Valuation-using-Python
This project predicts the valuation of cricket players based on their performance metrics and experience. It uses a machine learning model (Random Forest Regressor) trained on player statistics to estimate auction values.

Features<br>

Performance-based valuation using runs, matches, strike rate, and wickets.<br>

Experience adjustment to boost valuation for senior players.<br>

Minimum base price of ₹30 Lakh following IPL rules.<br>

No upper limit on valuation (dynamic based on player stats).<br>

Web-based UI with Streamlit for easy valuation lookup.<br>

Dataset<br>

Ensure the dataset realistic_merged_data.csv is present in the root directory. It should contain the following columns:<br>

Player, Year, Matches, Runs, Batting_Strike_Rate, Wickets_Taken<br>


 Install dependencies<br>

pip install -r requirements.txt<br>

 Train the Model (if needed)<br>

python train_model.py<br>

 Run the Streamlit App<br>

streamlit run player_valuation.py<br>

Usage<br>

Open the localhost URL from the Streamlit output in your browser.<br>

Select a player from the dropdown to view their valuation.<br>

Model considers past performance & experience to calculate valuation.<br>
