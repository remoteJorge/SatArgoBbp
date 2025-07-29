import os
import pickle

def export_model(model, setup):
    """
    Saves the trained model to disk following the experiment setup.

    Args:
        model: Trained model object
        setup: ExperimentSetup instance
    """
    path = f"../results/{setup.region}/{setup.depth}/{setup.experiment}/Model"
    filename = f"{path}/{setup.region}_{setup.depth}_{setup.experiment}_model.sav"

    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to: {filename}")
