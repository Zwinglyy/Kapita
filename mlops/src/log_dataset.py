import mlflow
import pandas as pd
from mlflow.data import from_pandas

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

# Retrieve the run information
logged_run = mlflow.get_run(run.info.run_id)

# Retrieve the Dataset object
logged_dataset = logged_run.inputs.dataset_inputs[0].dataset

# View some of the recorded Dataset information
print(f"Dataset name: {logged_dataset.name}")
print(f"Dataset digest: {logged_dataset.digest}")
print(f"Dataset profile: {logged_dataset.profile}")
print(f"Dataset schema: {logged_dataset.schema}")
