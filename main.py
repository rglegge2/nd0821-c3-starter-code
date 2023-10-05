import logging
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import cat_features

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

app = FastAPI()


# Declare the data object with its components and their type.
class InputData(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example='State-gov')
    fnlgt: int = Field(example=77516)
    education: str = Field(example='Bachelors')
    education_num: int = Field(example=13)
    marital_status: str = Field(example='Never-married')
    occupation: str = Field(example='Adm-clerical')
    relationship: str = Field(example='Not-in-family')
    race: str = Field(example='White')
    sex: str = Field(example='Male')
    capital_gain: int = Field(example=2174)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(example='United_States')


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
async def run_inference(input_data: InputData) -> dict:
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
