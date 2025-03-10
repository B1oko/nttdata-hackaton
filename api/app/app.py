from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import joblib
import pandas as pd
import numpy as np

MODEL_PKL = "model_nn_1.pkl"


# Cargar el modelo entrenado (asegúrate de que model.pkl está en la misma carpeta)
from model import ModeloNN
model = joblib.load(f"models/{MODEL_PKL}")

# Inicializar la API
app = FastAPI(title="Income Prediction API", version="1.0.0")

# Definir el esquema de entrada
class InputData(BaseModel):
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

class RequestModel(BaseModel):
    input_data: List[InputData]

# Definir el esquema de salida
class PredictionResponse(BaseModel):
    income: str
    confidence: float

class ResponseModel(BaseModel):
    predictions: List[PredictionResponse]

# Valores por defecto para datos faltantes
DEFAULT_VALUES = {
    "occupation": "Unknown",
}

# Función de preprocesamiento de datos
def preprocess_data(data: List[Dict]):
    df = pd.DataFrame(data)
    
    # Reemplazar valores faltantes (?)
    for column, default_value in DEFAULT_VALUES.items():
        df[column] = df[column].replace("?", default_value)

    # Codificación de variables categóricas (debe ser igual a como fue entrenado el modelo)
    categorical_features = ["education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
    df = pd.get_dummies(df, columns=categorical_features)

    # Asegurar que todas las columnas usadas en entrenamiento estén presentes
    model = joblib.load(f"models/{MODEL_PKL}")  # Cargar nombres de columnas usadas en el modelo
    for col in model:
        if col not in df.columns:
            df[col] = 0  # Agregar columnas faltantes con valor 0

    df = df[model]  # Asegurar orden correcto de las columnas
    return df

# Endpoint de predicción
@app.post("/predict", response_model=ResponseModel)
def predict(request: RequestModel):
    input_data = request.input_data
    processed_data = preprocess_data([data.dict() for data in input_data])

    # Realizar la predicción
    probabilities = model.predict_proba(processed_data)
    predictions = model.predict(processed_data)

    # Formatear la respuesta
    response = [
        {"income": "<=50K" if pred == 0 else ">50K", "confidence": float(np.max(prob))}
        for pred, prob in zip(predictions, probabilities)
    ]
    
    return {"predictions": response}

# Endpoint de salud
@app.get("/health")
def health():
    return {"status": "ok", "model_version": "1.0.0"}
