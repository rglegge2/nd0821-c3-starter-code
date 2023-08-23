import logging
import pickle

import numpy as np
import pandas as pd
import pytest

from starter.starter.ml import model
from starter.starter.ml.data import process_data
from starter.starter.ml.model import compute_model_metrics, model_slice_performance
from starter.starter.train_model import cat_features

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@pytest.fixture(scope="module")
def clf():
    return pickle.load(open("starter/model/trained_model.pkl", 'rb'))


@pytest.fixture(scope="module")
def encoder():
    return pickle.load(open("starter/model/trained_encoder.pkl", 'rb'))


@pytest.fixture(scope="module")
def lb():
    return pickle.load(open("starter/model/trained_lb.pkl", 'rb'))


@pytest.fixture(scope="module")
def X(encoder, lb):
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
    input_df = pd.DataFrame.from_dict(input_data, orient='index').T
    processed_input_data, _, _, _ = process_data(
        input_df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )
    return processed_input_data


@pytest.fixture(scope="module")
def preds():
    return np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1])


@pytest.fixture(scope="module")
def y_test():
    return np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1])


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("starter/data/census.csv")


def test_inference(clf, X):
    preds = model.inference(clf, X)
    assert isinstance(preds, np.ndarray)
    assert preds[0] == 0


def test_compute_model_metrics(y_test, preds):
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_model_slice_performance(df, clf, encoder, lb):
    results = model_slice_performance(df, "sex", clf, encoder, lb)
    assert (len(results)) == 2
    assert "precision" in results[0].keys()
    assert "recall" in results[0].keys()
    assert "fbeta" in results[0].keys()
    assert "class" in results[0].keys()
    assert "count" in results[0].keys()
    assert results[0]["feature"] == "sex"
