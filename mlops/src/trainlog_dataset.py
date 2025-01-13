import mlflow
import pandas as pd
from mlflow.data import from_pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")

dataset_source_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
raw_data = pd.read_csv(dataset_source_url, delimiter=";")

# Create an instance of a PandasDataset
dataset = from_pandas(
    raw_data, source=dataset_source_url, name="wine quality - white", targets="quality"
)

# Log the Dataset to an MLflow run by using the `log_input` API
with mlflow.start_run() as run:
    mlflow.log_input(dataset, context="training")

    # Split the data into training and test sets
    X = raw_data.drop(columns=["quality"])
    y = raw_data["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    mlflow.log_metric("logistic_regression_accuracy", lr_accuracy)

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    mlflow.log_metric("random_forest_accuracy", rf_accuracy)

    # Log models
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

# Retrieve the run information
logged_run = mlflow.get_run(run.info.run_id)

# Retrieve the Dataset object
logged_dataset = logged_run.inputs.dataset_inputs[0].dataset

# View some of the recorded Dataset information
print(f"Dataset name: {logged_dataset.name}")
print(f"Dataset digest: {logged_dataset.digest}")
print(f"Dataset profile: {logged_dataset.profile}")
print(f"Dataset schema: {logged_dataset.schema}")