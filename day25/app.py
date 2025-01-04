from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
import pickle
import numpy as np
from config.config import Config
from utils.logger import setup_logger
import numpy as np

logger = setup_logger('api')

class FeatureInput(BaseModel):
    LSTAT: float
    RM: float
    CRIM: float
    PTRATIO: float
    INDUS: float
    TAX: float
    NOX: float
    B: float

app = FastAPI()

# Load model at startup
try:
    with open(Config.MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.post("/predict")
async def predict(features: FeatureInput):
    try:
        feature_dict = features.dict()
        input_df = pd.DataFrame([feature_dict])
        prediction = model.predict(input_df)
        
        logger.info(f"Prediction made for input: {feature_dict}")
        return {"prediction": float(np.exp(prediction[0]))}
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)