import streamlit as st
import pandas as pd
from model import get_trained_model

# Load the trained model.
model = get_trained_model()

st.title("🏈 AI Offensive Coordinator")
st.write("Enter the current game situation below to get an instant play-calling recommendation based on machine learning.")

# Create input widgets in the sidebar.
st.sidebar.header("Game Situation Inputs")
down = st.sidebar.selectbox("Down", [1, 2, 3, 4])
distance = st.sidebar.slider("Distance to Go (Yards)", 1, 20, 5)
yardline = st.sidebar.slider("Field Position (Yards from Own Endzone)", 1, 99, 50)
score_diff = st.sidebar.slider("Score Differential (Negative = Trailing, Positive = Leading)", -28, 28, 0)

# Trigger recommendation.
if st.button("Get Play Recommendation"):
	input_data = pd.DataFrame({
		"down": [down],
		"distance": [distance],
		"yardline": [yardline],
		"score_diff": [score_diff],
	})

	prediction = model.predict(input_data)[0]
	probabilities = model.predict_proba(input_data)[0]

	play_name = "PASS 🏈" if prediction == 1 else "RUN 🏃"
	run_prob = probabilities[0] * 100
	pass_prob = probabilities[1] * 100

	st.subheader("Play Recommendation Result:")
	st.success(f"**Recommended Call: {play_name}**")

	st.write("### Model Confidence Breakdown:")
	st.write(f"* **Run Probability:** `{run_prob:.1f}%`")
	st.write(f"* **Pass Probability:** `{pass_prob:.1f}%`")
