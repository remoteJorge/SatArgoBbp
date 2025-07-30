import pandas as pd

def load_metrics(region, experiment, depth):
    path = f"../results/{region}/{depth}/{experiment}/Model/Metrics/{region}_{depth}_{experiment}_depth_metrics.csv"
    df = pd.read_csv(path, index_col=0, header=None).iloc[1:, :]
    return df

def load_all_metrics(regions, experiments, depths):
    metrics_dict = {"R2": pd.DataFrame(), "MAE": pd.DataFrame(), "Bias": pd.DataFrame()}

    for region in regions:
        for depth in depths:
            for exp in experiments:
                df = load_metrics(region, exp, depth)
                for metric in metrics_dict:
                    s = df.loc[metric]
                    s.name = f"{region}_{exp}"
                    metrics_dict[metric] = pd.concat([metrics_dict[metric], s], axis=1)

    return metrics_dict["R2"], metrics_dict["MAE"], metrics_dict["Bias"]
