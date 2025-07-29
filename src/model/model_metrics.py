import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,root_mean_squared_error


def median_absolute_percentage_deviation(y_true, y_pred, multioutput=None):
    """
    Compute the Median Absolute Percentage Deviation (MAPD) for single or multi-output models.

    Parameters:
    y_true (array-like): Ground truth values, shape (n_samples,) or (n_samples, n_outputs).
    y_pred (array-like): Predicted values, shape (n_samples,) or (n_samples, n_outputs).
    multioutput (str, optional): Defines aggregation (only used for multi-output cases):
        - None (default): Computes a single MAPD value for all outputs.
        - 'raw_values': Returns MAPD per output.
        - 'uniform_average': Returns the average MAPD across all outputs.

    Returns:
    float or ndarray: MAPD value(s).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)  # Ensure arrays

    # Compute MAPD for each output (works for both single and multi-output cases)
    mapd_values = np.median(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8), axis=0) * 100

    # If single-output (1D arrays), return a scalar MAPD value
    if y_true.ndim == 1:
        return mapd_values.item()  # Convert single-value array to float

    # If multi-output and `multioutput` is specified, apply aggregation
    if multioutput == "raw_values":
        return mapd_values  # Return MAPD for each output separately
    elif multioutput == "uniform_average":
        return np.mean(mapd_values)  # Return average MAPD across all outputs
    else:
        return np.median(mapd_values)  # Default: single MAPD value across all outputs
       

def overall_metrics(y_true, y_pred):
    return {
        "R2": r2_score(np.log10(y_true), np.log10(y_pred)),  
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPD": median_absolute_percentage_deviation(y_true, y_pred)
    }


def save_overall_metrics(metrics_dict, setup):
    df = pd.DataFrame({
        "Experiment": [setup.experiment] + [""] * (len(metrics_dict) - 1),
        "Depth": [f"{setup.depth}m"] + [""] * (len(metrics_dict) - 1),
        "Metric": list(metrics_dict.keys()),
        "Value": [f"{val:.4e}" if isinstance(val, float) and val < 0.01 else f"{val:.2f}" for val in metrics_dict.values()]
    })

    path = f"../results/{setup.region}/{setup.depth}/{setup.experiment}/Model/Metrics"
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/{setup.region}_{setup.depth}_{setup.experiment}_model_overall_metrics.csv", index=False)
    print(f"Saved overall metrics to {path}")
    
def depth_metrics(y_true, y_pred, setup):
    """
    Compute depth-resolved metrics and return as a formatted DataFrame.

    Args:
        y_true (array): Ground truth values.
        y_pred (array): Predicted values.

    Returns:
        pd.DataFrame: Metrics indexed by metric name, with one column per depth level (X1, X2, ...).
    """
    metrics = {
        "R2": r2_score(np.log10(y_true),np.log10( y_pred), multioutput='raw_values'),
        "MSE": mean_squared_error(y_true, y_pred, multioutput='raw_values'),
        "MAE": mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
        "RMSE": root_mean_squared_error(y_true, y_pred, multioutput='raw_values'),
        "MAPD": median_absolute_percentage_deviation(y_true, y_pred, multioutput='raw_values'),
        "Bias": np.mean(y_pred - y_true, axis=0)
    }
    
    depth_max = int(setup.depth)
    depth_range = depth_range = [-i for i in range(1, int(setup.depth) + 2, 2)]

    df = pd.DataFrame.from_dict(metrics, orient="index")
    return df
    
def overall_validation_metrics(y_true, y_pred):
    return {
        "R2": r2_score(np.log10(y_true), np.log10(y_pred)),  
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPD": median_absolute_percentage_deviation(y_true, y_pred)
    }
    df.columns = depth_range
    df.index.name = "Metric"

    return df
    
def save_overall_validation_metrics(y_val_bbp, y_pred_val_bbp, setup, export_config):
    """
    Saves overall validation metrics to a CSV file.

    Parameters:
    - y_val_bbp, y_pred_val_bbp: true and predicted values (already in linear Bbp space)
    - setup: ExperimentSetup object
    - export_config: ExportConfig object (should have export_validation_metrics = True)
    """

    # Build DataFrame
    df = pd.DataFrame({
        "Experiment": [f"{setup.experiment}"] + [""] * 4,
        "Depth": [f"{setup.depth}m"] + [""] * 4,
        "Metric": ["R2", "MSE", "MAE", "RMSE", "MAPD"],
        "Value": [
            f"{r2_score(np.log10(y_val_bbp), np.log10(y_pred_val_bbp)):.2f}",
            f"{mean_squared_error(y_val_bbp, y_pred_val_bbp):.2e}",
            f"{mean_absolute_error(y_val_bbp, y_pred_val_bbp):.2e}",
            f"{root_mean_squared_error(y_val_bbp, y_pred_val_bbp):.2e}",
            f"{median_absolute_percentage_deviation(y_val_bbp, y_pred_val_bbp):.2f}"
        ]
    })

    if not getattr(export_config, "export_validation_metrics", False):
        return df

    # Save to disk
    out_path = f"../Results/{setup.region}/{setup.depth}/{setup.experiment}/Validation/Metrics"
    os.makedirs(out_path, exist_ok=True)
    csv_path = os.path.join(out_path, f"{setup.region}_{setup.depth}_{setup.experiment}_Val_overall_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved overall validation metrics to: {csv_path}")
    
def depth_val_metrics(y_true, y_pred, setup):
    """
    Compute depth-resolved metrics and return as a formatted DataFrame.

    Args:
        y_true (array): Ground truth values.
        y_pred (array): Predicted values.

    Returns:
        pd.DataFrame: Metrics indexed by metric name, with one column per depth level (X1, X2, ...).
    """
    metrics = {
        "R2": r2_score(np.log10(y_true), np.log10(y_pred), multioutput='raw_values'),
        "MSE": mean_squared_error(y_true, y_pred, multioutput='raw_values'),
        "MAE": mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
        "RMSE": root_mean_squared_error(y_true, y_pred, multioutput='raw_values'),
        "MAPD": median_absolute_percentage_deviation(y_true, y_pred, multioutput='raw_values'),
        "Bias": np.mean(y_pred - y_true, axis=0)
    }
    
    depth_max = int(setup.depth)
    depth_range = depth_range = [-i for i in range(1, int(setup.depth) + 2, 2)]

    df = pd.DataFrame.from_dict(metrics, orient="index")
    
    return df
