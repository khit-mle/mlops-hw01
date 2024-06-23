import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict-iris-species")
async def predict_iris_species(features: IrisFeatures):
    try:
        # Create a DataFrame with the appropriate column names
        input_data = pd.DataFrame(
            [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]],
            columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        )

        # Transform the input data using the scaler
        scaled_data = scaler.transform(input_data)

        # Predict the species using the model
        prediction = model.predict(scaled_data)

        return {"species": prediction[0]}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
