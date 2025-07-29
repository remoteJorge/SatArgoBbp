import numpy as np

def select_features(dataset, setup):
    """
    Selects input features and output targets based on the experiment and depth.

    Args:
        dataset (pd.DataFrame): The dataset.
        setup (ExperimentSetup): The experiment config.

    Returns:
        tuple: (features, bbp_outputs)
    """
    # Input feature blocks
    st = dataset.columns[7:10]             # Spatio-temporal
    mld_sla = dataset.columns[14:16]       # MLD + SLA
    par = dataset.columns[16]              # PAR
    gcolor = dataset.columns[16:22]        # GlobColor
    pcas = dataset.columns[70:90]          # PCA
    iops = dataset.columns[22:30]          # IOPs
    s3 = dataset.columns[58:70]            # Sentinel-3

    # Output targets depend on depth
    if setup.depth == '50':
        bbp_outputs = dataset.columns[90:]
    elif setup.depth == '250':
        bbp_outputs = dataset.columns[92:]
    else:
        raise ValueError(f"Unsupported depth: {setup.depth}")

    # Experiment-specific feature combinations
    experiment_features = {
        "GCGOBGC": np.r_[st, mld_sla, gcolor, pcas],
        "S3OLCIBGC": np.r_[st, mld_sla, s3, pcas],
        "S3IOPS": np.r_[st, mld_sla, s3, iops],
        "S3OLCI": np.r_[st, mld_sla, s3]
    }

    if setup.experiment not in experiment_features:
        raise ValueError(f"Unknown experiment: {setup.experiment}")

    return experiment_features[setup.experiment], bbp_outputs
