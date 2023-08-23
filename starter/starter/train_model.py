# Script to train machine learning model.
import logging
import os
import pickle

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, compute_model_metrics, inference, model_slice_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def compute_model_slice_performance(df, model, encoder, lb):
    """
    Computes the model performance on slices of all categorical features and outputs results to slice_output.txt

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to run the performance metrics against. Preferably the test dataset.
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer
    """
    all_results = []
    for feature in cat_features:
        feature_results = model_slice_performance(df, feature, model, encoder, lb)
        all_results += feature_results
    all_res_df = pd.DataFrame(all_results)
    all_res_df.to_csv("slice_output.txt", index=False)


def main():
    # Load in data
    data_path = '../data/census.csv'
    data = pd.read_csv(data_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, stratify=data['salary'], test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Load existing model or train new model
    model_path = "../model/trained_model.pkl"
    encoder_path = "../model/trained_encoder.pkl"
    lb_path = "../model/trained_lb.pkl"
    if os.path.exists(model_path):
        logger.info(f"Model exists at: {model_path}. Loading existing model.")
        model = pickle.load(open(model_path, "rb"))
    else:
        logger.info("Model path does not exist. Training new model.")
        model = train_model(X_train, y_train)
        logger.info(f"Saving model to: {model_path}")
        pickle.dump(model, open(model_path, "wb"))
        pickle.dump(encoder, open(encoder_path, "wb"))
        pickle.dump(lb, open(lb_path, "wb"))
        logger.info(f"Model saved successfully to: {model_path}")

    # Test model performance
    logger.info("Calculating performance metrics...")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logger.info(f"Precision: {precision}\nRecall: {recall}\nF-Score: {fbeta}")

    cm = confusion_matrix(y_test, preds)
    logger.info(f"Confusion Matrix:\n{cm}")

    logger.info("Computing model performance on feature slices")
    compute_model_slice_performance(test, model, encoder, lb)
    logger.info("Feature slice performance saved to slice_output.txt")


if __name__ == '__main__':
    main()
