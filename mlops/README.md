# MLOps Practicum: Model Logging and Tracking with MLflow

This practicum focuses on using MLflow for experiment tracking and model logging LOCALLY. You'll learn how to log different types of ML models, compare their performance, and track datasets using MLflow.

## Prerequisites
- Python 3.9 or higher
- Git
- Basic understanding of Machine Learning concepts
- Basic understanding of command line operations

## Setup Instructions

### 1. Clone the Repository
```bash
# Clone this repository
git clone https://github.com/hani-ramadhan/mlops.git
cd mlops

# Create project directories if they don't exist
mkdir -p data models
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .

# Activate virtual environment
# For Windows:
mlops\Scripts\activate
# For Unix or MacOS:
source mlops/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

## Practicum Tasks

### Task 1: Model Comparison with Synthetic Data

1. Start MLflow UI server:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

2. Generate synthetic data:
```bash
python data_generator.py
```
This will create:
- `data/train.csv`: Initial training data
- `data/new_data.csv`: Data with drift for later use

3. Train and log models:
```bash
python train_model.py
```
This will:
- Train a Random Forest model
- Train a Logistic Regression model
- Log both models with their metrics to MLflow

4. Access MLflow UI:
- Open your browser and navigate to `http://127.0.0.1:5000`
- Compare the models by:
  - Checking accuracy and AUC scores
  - Looking at model parameters
  - Examining run metadata

5. Required Screenshot:
   - MLflow experiment page showing both models
   - Comparison view of Random Forest vs Logistic Regression metrics
   - Model parameters page for both models

### Task 2: Dataset Logging with Your Own Data

1. Choose a dataset:
   - Find an interesting dataset from Kaggle, UCI Repository, or other sources
   - Recommended: Classification datasets with 5-15 features

2. Create a new Python script `log_dataset.py` that logs the dataset as the input of your ML task. Check the tutorial from mlflow [here](https://mlflow.org/docs/latest/tracking/data-api.html)



3. Run dataset logging:
```bash
python log_dataset.py
```

4. Required Screenshot:
   - MLflow artifacts showing your logged dataset
   - Dataset statistics in MLflow UI
   - Dataset info page

### Task 3: Model Comparison with Your Dataset

1. Create a new training script for your dataset (you can refer the original script) to log the new dataset with new training procedure. Use three different models, for example: logistic regression, random forest, neural network.

2. Train models on your dataset:
```bash
python train_my_dataset.py
```

3. Required Screenshot:
   - MLflow comparison view of models trained on your dataset
   - Model parameters and metrics for each model
   - Any interesting findings or patterns you discovered

### Task 4: Drift Detection with MLFlow Integration

1. Understand PSI Calculation
- Study the PSI formula and interpretation:
  * PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
  * Thresholds: <0.1 (no drift), 0.1-0.2 (moderate), >0.2 (high drift)

2. Implement Drift Detection

3. Run Drift Detection:

```bash
python detect_drift.py
```

4. Required Screenshots:
- MLflow experiment page showing drift detection runs
- PSI values for each feature
- Distribution comparisons in artifacts
- Overall drift analysis results

5. Experiment with Different Drift Levels:
Modify data_generator.py drift parameters to create:
- No drift (PSI < 0.1)
- Moderate drift (0.1 ≤ PSI < 0.2)
- High drift (PSI ≥ 0.2)

## Submission Requirements

Create a report (README.md) in your repository containing:

1. **Environment Setup**
   - Screenshot of successful MLflow UI startup
   - List of installed packages (requirements.txt)

2. **Task 1: Synthetic Data Results**
   - Screenshots of model comparisons
   - Brief analysis of which model performed better and why
   - Any interesting patterns in the metrics

3. **Task 2: Dataset Documentation**
   - Description of your chosen dataset
   - Screenshots of dataset logging in MLflow
   - Any challenges faced during dataset logging

4. **Task 3: Custom Dataset Results**
   - Screenshots of model comparisons
   - Analysis of model performance
   - Comparison with synthetic data results

5. **Task 4: Drift Detection**
   - Code implementation in your repository
   - Screenshots of MLflow tracking for drift detection
   - Analysis of drift detection results
   - Document different drift parameter setting based on the data generator

7. **Reflection** (half A4 page report)
   - What you learned about MLflow
   - Challenges faced and how you overcame them
   - Suggestions for improvement

8. **Your Github repo**
   - Denote your full name and student ID number clearly with the updated codes and created model, mlflow logging, and data directories

## Troubleshooting

Common issues and solutions:

1. MLflow UI not starting:
   - Check if port 5000 is available
   - Ensure MLflow is installed correctly
   - Try a different port

2. Model logging fails:
   - Check file paths
   - Verify data format
   - Look for missing dependencies

3. Dataset logging issues:
   - Check file permissions
   - Verify data types
   - Ensure consistent column names

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Dataset Sources](https://www.kaggle.com/datasets)
- [Git Tutorial](https://git-scm.com/docs/gittutorial)

## Next Steps

After completing this practicum, you can:
1. Experiment with different ML models
2. Try more complex datasets
3. Explore MLflow's model registry
4. Add automated testing
