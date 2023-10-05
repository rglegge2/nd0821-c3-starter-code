import logging
import pickle

import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from starter.ml.model import inference
from starter.ml.data import process_data
from starter.train_model import cat_features

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Declare the data object with its components and their type.
class InputData(BaseModel):
    age: int = Field(examples=[39])
    workclass: str
    fnlgt: int
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


app = FastAPI()
model_path = "model/trained_model.pkl"
encoder_path = "model/trained_encoder.pkl"
lb_path = "model/trained_lb.pkl"
model = pickle.load(open(model_path, "rb"))
encoder = pickle.load(open(encoder_path, "rb"))
lb = pickle.load(open(lb_path, "rb"))


@app.get("/")
async def hello_world():
    return "Hello World!"


@app.post("/inference")
async def run_inference(input_data: Annotated[InputData, Body(examples=[{
                    'age': 39,
                    'workclass': 'State-gov',
                    'fnlgt': 77516,
                    'education': 'Bachelors',
                    'education_num': 13,
                    'marital_status': 'Never-married',
                    'occupation': 'Adm-clerical',
                    'relationship': 'Not-in-family',
                    'race': 'White',
                    'sex': 'Male',
                    'capital_gain': 2174,
                    'capital_loss': 0,
                    'hours_per_week': 40,
                    'native_country': 'United_States'
                }])]) -> dict:
    input_dict = input_data.dict()
    logger.info(input_dict)
    input_df = pd.DataFrame.from_dict(input_dict, orient='index').T
    processed_input_data, _, _, _ = process_data(
        input_df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )
    pred = inference(model, processed_input_data)
    if pred[0] < 0.5:
        pred_output = '<=50K'
    else:
        pred_output = '>50k'
    return {"Prediction": pred_output}
