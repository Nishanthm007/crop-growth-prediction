import joblib
import numpy as np

model = joblib.load('../model/crop_model.pkl')

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    result = model.predict(input_data)
    return result[0]

# Example usage
# print(predict_crop(90, 42, 43, 20.0, 80.0, 6.5, 100.0))
