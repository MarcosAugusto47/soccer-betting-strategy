# mlflow_setup.py
import mlflow

def setup_experiment(experiment_name):
    """
    Sets up the MLflow experiment.

    Parameters:
    - experiment_name: Name of the experiment (e.g., the type of optimization)
    """
    mlflow.set_experiment(experiment_name)
