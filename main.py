from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from tensorflow.keras.models import load_model
from keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load the trained model and scaler
model = load_model('model/house_price_model.h5')
scaler = joblib.load('model/scaler.pkl')

# Define a model for the input data (you can adjust this according to your data)
class HouseData(BaseModel):
    squareMeters: float
    numberOfRooms: float
    hasYard: float
    hasPool: float
    floors: float
    numPrevOwners: float
    made: float
    isNewBuilt: float
    hasStormProtector: float
    basement: float
    attic: float
    garage: float
    hasStorageRoom: float
    hasGuestRoom: float
    hasFireplace: float 
# Endpoint for predictions
@app.post("/predict")
def predict(data: HouseData):
    try:
        # Convert the input data to a format the model expects (numpy array)
        input_data = np.array([[data.squareMeters, data.numberOfRooms, data.hasYard, 
                                data.hasPool, data.floors, data.numPrevOwners, 
                                data.made, data.isNewBuilt, data.hasStormProtector,
                                data.basement, data.attic, data.garage, 
                                data.hasStorageRoom, data.hasGuestRoom]])
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        # Return the prediction
        return {"prediction": int(prediction[0])}  # Convert to int for clarity if categories are integers
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")


