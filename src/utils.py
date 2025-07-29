import os 
import numpy as np
from dataclasses import dataclass

def start_experiments():
    """
    Ensures the required folder structure exists for experiments.
    Creates missing folders and prints a status message.
    """
    # Define the base directory
    base_dir = "../results"

    # Define the full structure from your tree output
    structure = {
        "Comparisons": ["250", "50"],
        "NA": {
            "250": ["GCGOBGC", "S3IOPS", "S3OLCI", "S3OLCIBGC"],
            "50": ["GCGOBGC", "S3IOPS", "S3OLCI", "S3OLCIBGC"],
        },
        "STG": {
            "250": ["GCGOBGC", "S3IOPS", "S3OLCI", "S3OLCIBGC"],
            "50": ["GCGOBGC", "S3IOPS", "S3OLCI", "S3OLCIBGC"],
        },
        "Validation": []  # Standalone folder
    }

    # Variable to track if we created any folder
    folders_created = False

    # Function to check and create folders
    def ensure_folder_exists(folder_path):
        nonlocal folders_created  # Access the variable from the outer function
        if os.path.exists(folder_path):
            return  # Folder exists, do nothing
        os.makedirs(folder_path, exist_ok=True)
        folders_created = True  # Flag that a folder was created
        print(f"Created folder: {folder_path}")

    # Ensure all directories exist
    for main_folder, sub_items in structure.items():
        main_path = os.path.join(base_dir, main_folder)
        ensure_folder_exists(main_path)  # Create main folders

        if isinstance(sub_items, list):  # For Comparisons and Validation
            for sub in sub_items:
                ensure_folder_exists(os.path.join(main_path, sub))

        elif isinstance(sub_items, dict):  # For NA and STG with depths
            for depth, experiments in sub_items.items():
                depth_path = os.path.join(main_path, depth)
                ensure_folder_exists(depth_path)

                for experiment in experiments:
                    experiment_path = os.path.join(depth_path, experiment)
                    ensure_folder_exists(experiment_path)

                    # Create 'Model' and 'Validation' directories
                    model_path = os.path.join(experiment_path, "Model")
                    validation_path = os.path.join(experiment_path, "Validation")

                    ensure_folder_exists(model_path)
                    ensure_folder_exists(validation_path)

                    # Create 'Figures' and 'Metrics' inside 'Model' and 'Validation'
                    ensure_folder_exists(os.path.join(model_path, "Figures"))
                    ensure_folder_exists(os.path.join(model_path, "Metrics"))
                    ensure_folder_exists(os.path.join(validation_path, "Figures"))
                    ensure_folder_exists(os.path.join(validation_path, "Metrics"))

    # Print the final message
    if folders_created:
        print("All required folders have been created.")
    else:
        print("Folders are configured.")
        
@dataclass
class ExperimentSetup:
    region: str
    depth: str
    experiment: str

    # All options
    ALL_REGIONS = ["NA", "STG"]
    ALL_DEPTHS = ["50", "250"]
    ALL_EXPERIMENTS = ["GCGOBGC", "S3OLCIBGC", "S3IOPS", "S3OLCI"]

    VALIDATION_FLOATS = {
        "STG": {"50": 3902125, "250": 3902125},
        "NA": {"50": 6902545, "250": 6902671}
    }

    def validation_float(self) -> int:
        return self.VALIDATION_FLOATS[self.region][self.depth]

@dataclass
class ExportConfig:
    # Single experiment exports
    export_datasets: bool = False
    export_model: bool = False
    export_model_metrics: bool = True
    export_fig_depth: bool = False
    export_fig_profile: bool = False
    export_fig_importance: bool = False
    export_validation_metrics: bool = False
    export_fig_val: bool = False
    
def prepare_xy(df, features, targets):
    X = df[features]
    y = df[targets]
    ylog = np.log10(y)
    return X, ylog

def log10_to_bbp(y_log):
    return np.power(10, y_log)
