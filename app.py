from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load model and scaler
X_train = joblib.load("X_train_columns.pkl")
scaler = joblib.load("scaler.pkl")
loaded_model = joblib.load("best_model.pkl")

app = Flask(__name__)

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])

    # Categorical + Numeric features
    categorical_columns = ['gender', 'smoking_history']
    numerical_columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    # One-hot encode
    input_processed = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

    # Scale numericals
    input_processed[numerical_columns] = scaler.transform(input_processed[numerical_columns])

    # Add missing training columns
    for col in X_train.columns:
        if col not in input_processed.columns:
            input_processed[col] = 0

    # Ensure correct order
    input_processed = input_processed[X_train.columns]

    # Predict
    prediction = loaded_model.predict(input_processed)[0]
    probability = loaded_model.predict_proba(input_processed)[0][1]

    return prediction, probability

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    if request.method == 'POST':
        input_data = {
            'gender': request.form['gender'],
            'smoking_history': request.form['smoking_history'],
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'bmi': float(request.form['bmi']),
            'HbA1c_level': float(request.form['HbA1c_level']),
            'blood_glucose_level': float(request.form['blood_glucose_level']),
        }

        prediction, probability = make_prediction(input_data)

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)