import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_trained_model():
	# Create a synthetic dataset representing game situations.
	np.random.seed(42)
	n_samples = 1200

	down = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
	distance = np.random.randint(1, 15, size=n_samples)
	yardline = np.random.randint(1, 100, size=n_samples)
	score_diff = np.random.randint(-21, 22, size=n_samples)

	# Target: 0 = run, 1 = pass.
	target = []
	for current_down, yards_to_go in zip(down, distance):
		if current_down == 3 and yards_to_go >= 6:
			target.append(1)
		elif current_down == 1 or yards_to_go <= 2:
			target.append(0)
		else:
			target.append(np.random.choice([0, 1], p=[0.45, 0.55]))

	df = pd.DataFrame({
		"down": down,
		"distance": distance,
		"yardline": yardline,
		"score_diff": score_diff,
		"play_type": target,
	})

	X = df[["down", "distance", "yardline", "score_diff"]]
	y = df["play_type"]

	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X, y)

	return model


if __name__ == "__main__":
	model = get_trained_model()
	print("AI Offensive Coordinator model trained successfully!")
