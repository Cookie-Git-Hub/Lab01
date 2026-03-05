import joblib
import numpy as np

loaded_model = joblib.load('models/model_v1.joblib')

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = loaded_model.predict(sample)
print(f"Prediction for {sample}: {prediction}")