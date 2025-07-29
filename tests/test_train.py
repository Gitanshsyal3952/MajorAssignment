import joblib
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def test_data_loading():
    X, y = fetch_california_housing(return_X_y=True)
    assert X.shape[0] > 0 and y.shape[0] > 0

def test_model_instance():
    model = joblib.load("src/model.joblib")
    assert isinstance(model, LinearRegression)

def test_model_is_trained():
    model = joblib.load("src/model.joblib")
    assert hasattr(model, "coef_")

def test_r2_threshold():
    X, y = fetch_california_housing(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
    model = joblib.load("src/model.joblib")
    y_pred = model.predict(X_test)
    assert r2_score(y_test, y_pred) > 0.5
