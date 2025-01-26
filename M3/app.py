# ## Group Name: MLOPS Group 63

# ## Group Member Names:
# 1.   Sivakumar G - 2023aa05486
# 2.   Pabbisetty Jayakrishna - 2023aa05487
# 3.   Ravi shankar S - 2023aa05488
# 4.   Srivatsan V R - 2023aa05962

# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('best_rf_model.joblib')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the request
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    
    # Make a prediction using the model
    prediction = model.predict(features)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
