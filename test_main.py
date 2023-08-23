import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_hello_world():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello World!"


def test_run_inference_success_0():
    input_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    data = json.dumps(input_data)
    r = client.post("/inference", data=data)
    assert r.status_code == 200
    assert r.json()['Prediction'] == "<=50K"


def test_run_inference_success_1():
    input_data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    data = json.dumps(input_data)
    r = client.post("/inference", data=data)
    assert r.status_code == 200
    assert r.json()['Prediction'] == ">50k"


def test_run_inference_fail():
    input_data = {
        "bad": "data"
    }
    data = json.dumps(input_data)
    r = client.post("/inference", data=data)
    assert r.status_code == 422
