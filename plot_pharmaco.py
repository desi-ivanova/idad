import os
import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns

import torch
import pyro

import mlflow

from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from estimators.mi import PriorContrastiveEstimation, NestedMonteCarloEstimation


def plot_designs_run(experiment_id, run_id):
    artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"

    # load hisotry and plot
    yy = "xi_0"
    temp = pd.read_csv(f"{artifact_path}/designs/designs_hist.csv",)
    plot = sns.lineplot(x="step", y=yy, data=temp, hue="order")
    plot.figure.savefig(f"mlflow_outputs/designs_hist.png")
    plot.figure.clf()

    pk_model = mlflow.pytorch.load_model(model_location, map_location="cpu")
    ## plot optimal designs for different thetas at the end of training
    dfs = []

    for i in range(10):
        obsdf = pk_model.eval(n_trace=1, verbose=False)[0]
        obsdf["theta_id"] = i
        dfs.append(obsdf.drop("observations", axis=1))

    dfs = pd.concat(dfs)
    plot = sns.scatterplot(x="xi_0", y="order", data=dfs, hue="theta_id")
    plot.figure.savefig(f"mlflow_outputs/final_designs.png")
    plot.figure.clf()

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
        # store images
        mlflow.log_param("plot", "Done")
        mlflow.log_artifact(f"mlflow_outputs/designs_hist.png", artifact_path="designs")
        mlflow.log_artifact(
            f"mlflow_outputs/final_designs.png", artifact_path="designs",
        )

    return


def plot_designs_experiment(experiment_id):
    filter_string = "params.status='complete'"
    meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)
    # run those that haven't yet been evaluated
    meta = [m for m in meta if "plot" not in m.data.params.keys()]
    experiment_run_ids = [run.info.run_id for run in meta]
    print(experiment_run_ids)
    for i, run_id in enumerate(experiment_run_ids):
        print(f"Plotting run {i+1} out of {len(experiment_run_ids)} runs...")
        plot_designs_run(experiment_id=experiment_id, run_id=run_id)
        print("\n")


if __name__ == "__main__":
    ## load data for plotting
    parser = argparse.ArgumentParser(description="Deep Adaptive Design: PK plots")
    parser.add_argument("--experiment-id", type=str)
    args = parser.parse_args()
    plot_designs_experiment(args.experiment_id)
