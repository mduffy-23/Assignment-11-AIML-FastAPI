# Packages
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import psycopg2

class SalesLagFeatures(BaseModel):
    Sales_M_1: float
    Sales_M_2: float
    Sales_M_3: float
    Sales_M_4: float
    Sales_M_5: float
    Sales_M_6: float

model = xgb.Booster()
model.load_model("model/model.json")

feature_names = [
    "Sales_M_1",
    "Sales_M_2",
    "Sales_M_3",
    "Sales_M_4",
    "Sales_M_5",
    "Sales_M_6"
]

# Database
conn = psycopg2.connect('postgresql://assignment11modellogs_user:MQm5DZXgI9Z0J7TkYJreKY8GreBawbQo@dpg-d4cdrpjipnbc739di4gg-a.ohio-postgres.render.com/assignment11modellogs')
cursor = conn.cursor()

def log_prediction(input_data, output_data):
    cursor.execute(
        "INSERT INTO predictions (M1, M2, M3, M4, M5, M6, Prediction) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (*input_data, output_data)
    )
    conn.commit()

# FastAPI
## Create FastAPI instance
app = FastAPI()

## Define prediction endpoint
@app.post("/predict")
def predict(data: SalesLagFeatures):
    input_data = np.array([
        data.Sales_M_1,
        data.Sales_M_2,
        data.Sales_M_3,
        data.Sales_M_4,
        data.Sales_M_5,
        data.Sales_M_6
    ]).reshape(1, -1)

    dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)
    prediction = float(model.predict(dmatrix)[0])

    log_prediction(input_data[0], prediction)

    return {"prediction": round(prediction, 2)}

# Mount Frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")