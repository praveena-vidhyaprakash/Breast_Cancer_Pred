from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
model = pickle.load(open('model.pkl', 'rb'))

# Use all 30 features from the breast cancer dataset
FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [float(request.form[feature]) for feature in FEATURES]
        data = np.array(inputs).reshape(1, -1)

        prediction = model.predict(data)[0]
        result = "Malignant" if prediction == 0 else "Benign"

        return render_template('index.html', features=FEATURES, result=result)
    except Exception as e:
        return render_template('index.html', features=FEATURES, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
