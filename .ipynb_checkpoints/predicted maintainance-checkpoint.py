import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load trained model and scaler
model = joblib.load("atm_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI
app = FastAPI(title="ATM Predictive Maintenance API", version="1.0")

class FeaturesInput(BaseModel):
    features: list  # Expecting a list of numerical values

@app.post("/predict")
def predict(data: FeaturesInput):
    try:
        expected_features = 9  # Make sure this matches model training
        if len(data.features) != expected_features:
            raise HTTPException(status_code=400, detail=f"Expected {expected_features} features, but got {len(data.features)}")

        # Convert and scale input
        input_data = np.array(data.features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Convert to readable label
        prediction_label = "No Failure" if prediction == 0 else "Failure"

        return {
            "prediction": int(prediction),
            "prediction_label": prediction_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
