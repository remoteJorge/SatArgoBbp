import pandas as pd
import glob 
import os

def load_dataset(setup):
    """
    Loads and prepares dataset for a given ExperimentSetup object.
    Returns train and validation splits.
    """
    file_pattern = f"../datasets/raw/Dataset_{setup.region}_{setup.depth}.txt"
    file = glob.glob(file_pattern)[0]
    dataset = pd.read_csv(file, sep="\t", na_values='NaN')

    dataset = dataset.rename(columns=lambda x: x.replace("Spy_", "Spi_") if x.startswith("Spy_") else x)
    dataset = dataset.rename(columns={"sla": "SLA"})

    cols = list(dataset.columns)
    par_idx, mld_idx = cols.index("PAR"), cols.index("MLD")
    cols[par_idx], cols[mld_idx] = cols[mld_idx], cols[par_idx]
    dataset = dataset[cols]

    val_wmo = setup.validation_float()
    validation = dataset[dataset["wmo"] == val_wmo]
    train = dataset[dataset["wmo"] != val_wmo]

    return train, validation

def export_dataset(train, validation, setup):
    """
    Exports train and validation datasets to disk.
    Creates required folders if they don't exist.
    """
    base_dir = "../datasets"
    train_dir = os.path.join(base_dir, "training")
    val_dir = os.path.join(base_dir, "validation")

    # Ensure folders exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Build file paths
    train_path = os.path.join(train_dir, f"Train_{setup.region}_{setup.depth}_{setup.experiment}.txt")
    val_path = os.path.join(val_dir, f"Validation_{setup.region}_{setup.depth}_{setup.experiment}.txt")

    # Export
    train.to_csv(train_path, sep="\t", index=False)
    validation.to_csv(val_path, sep="\t", index=False)

    print(f"Data exported to: {train_path} and {val_path}")
