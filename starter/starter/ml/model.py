import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from starter.starter.ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Duplicating this here because the project structure is annoying to deal with
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

# NOTE: Commenting this out and using default hyperparameters to reduce model size.
# Optional: implement hyperparameter tuning.
# def train_model(X_train, y_train):
#     """
#     Trains a machine learning model and returns it.
#
#     Inputs
#     ------
#     X_train : np.array
#         Training data.
#     y_train : np.array
#         Labels.
#     Returns
#     -------
#     model
#         Trained machine learning model.
#     """
#     logger.info("Training RandomForestClassifier")
#     rfc = RandomForestClassifier()
#     # Set up GridSearch for hyperparameter optimization
#     param_grid = {
#         'n_estimators': [100, 200, 500],
#         'max_depth': [10, 50, 100],
#         'criterion': ['gini', 'entropy']
#     }
#     cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='f1', cv=5, verbose=10)
#     cv_rfc.fit(X_train, y_train)
#     best_params = cv_rfc.best_params_
#     logger.info("Training complete!")
#     logger.info(f"Best Params: {best_params}")
#     return cv_rfc


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    logger.info("Training RandomForestClassifier")
    rfc = RandomForestClassifier(verbose=10)
    rfc.fit(X_train, y_train)
    logger.info("Training complete!")
    return rfc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    logger.info(f"Running model on {len(X)} rows")
    preds = model.predict(X)
    return preds


def model_slice_performance(df, feature, model, encoder, lb):
    """
    Calculate performance metrics on slices for a given feature

    Inputs
    ------
    df: pd.DataFrame
        DataFrame containing features to be used for slices
    feature: str
        The feature the slice will be performed on
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer

    Returns
    ------
    results: List[dict]
        Each dict within the list represents the model performance for each class for a given feature slice
        Performance metrics are precision, recall, and fbeta.
    """
    results = []
    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]
        X_processed, y_processed, _, _ = process_data(
            df_temp, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        preds = inference(model, X_processed)
        precision, recall, fbeta = compute_model_metrics(y_processed, preds)
        results.append({
            "feature": feature,
            "class": cls,
            "count": len(preds),
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta
        })
    return results
