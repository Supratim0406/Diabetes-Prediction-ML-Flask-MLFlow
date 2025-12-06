from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib

# Load ML model and preprocessing files
X_train = joblib.load("X_train_columns.pkl")
scaler = joblib.load("scaler.pkl")
loaded_model = joblib.load("best_model.pkl")

# FastAPI app
app = FastAPI()

# Templates directory
templates = Jinja2Templates(directory="templates")

# Prediction function
def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])

    categorical_columns = ['gender', 'smoking_history']
    numerical_columns = [
        'age', 'hypertension', 'heart_disease',
        'bmi', 'HbA1c_level', 'blood_glucose_level'
    ]

    # One-hot encode
    input_processed = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

    # Scale numeric columns
    input_processed[numerical_columns] = scaler.transform(input_processed[numerical_columns])

    # Add missing columns
    for col in X_train.columns:
        if col not in input_processed.columns:
            input_processed[col] = 0

    # Correct column order
    input_processed = input_processed[X_train.columns]

    # Predict
    prediction = loaded_model.predict(input_processed)[0]
    probability = loaded_model.predict_proba(input_processed)[0][1]

    return prediction, probability

# GET Route (Home Page)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST Route (Prediction Request)
@app.post("/", response_class=HTMLResponse)
async def predict(request: Request,
                  gender: str = Form(...),
                  smoking_history: str = Form(...),
                  age: float = Form(...),
                  hypertension: int = Form(...),
                  heart_disease: int = Form(...),
                  bmi: float = Form(...),
                  HbA1c_level: float = Form(...),
                  blood_glucose_level: float = Form(...)
                  ):

    # Preparing input dictionary
    input_data = {
        "gender": gender,
        "smoking_history": smoking_history,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }
    # Call prediction function
    prediction, probability = make_prediction(input_data)

    #Return updated HTML with prediction
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "probability": probability
        }
    )


# Running in production (Railway/Render)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
