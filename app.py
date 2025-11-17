import numpy as np
import pandas as pd
import joblib


from fastapi import FastAPI


app = FastAPI()

# Load any saved models
regression_model = joblib.load("models/regression_model.pkl")
# Define endpoints
@app.get("/")
def root():
    return {"message": "ML API is running!"}

@app.post("/predict_regression")
def predict_regression(data: dict):
    # example: expects features as a dictionary
    X = np.array([list(data.values())])
    pred = regression_model.predict(X)
    return {"prediction": int(pred[0])}

# Run the app with: uvicorn runCode:app --host
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
