import logging
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.model import inference
from starter.ml.data import process_data
from starter.train_model import cat_features

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Declare the data object with its components and their type.
class InputData(BaseModel):
    age: int = Field(examples=[39])
    workclass: str = Field(examples=['Private', 'State-gov', 'Federal-gov', 'Self-emp-not-inc', 'Self-emp-inc', 'Local-gov', 'Without-pay', 'Never-worked'])
    fnlgt: int = Field(examples=[77516])
    education: str = Field(examples=['Preschool', 'HS-grad', 'Some-college', 'Bachelors', 'Prof-school', 'Assoc-voc', 'Assoc-acdm', 'Masters', 'Doctorate'])
    education_num: int = Field(examples=[13])
    marital_status: str = Field(examples=['Married-civ-spouse', 'Never-married', 'Married-spouse-absent', 'Divorced', 'Separated', 'Widowed', 'Married-AF-spouse'])
    occupation: str = Field(examples=['Farming-fishing', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Other-service', 'Sales', 'Handlers-cleaners', 'Tech-support', 'Prof-specialty', 'Machine-op-inspct', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relationship: str = Field(examples=['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
    race: str = Field(examples=['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    sex: str = Field(examples=['Male', 'Female'])
    capital_gain: int = Field(examples=[2174])
    capital_loss: int = Field(examples=[0])
    hours_per_week: int = Field(examples=[40])
    native_country: str = Field(examples=['United-States', 'Cuba', 'Italy', 'Canada', 'Mexico', 'Jamaica', 'El-Salvador'])

    class Config:
        schema_extra = {
            "examples": [
                {
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
                }
            ]
        }


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
