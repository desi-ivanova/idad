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


def plot_designs_run(experiment_id, run_id, file_prefix, simdata):
    artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"

    # load hisotry and plot
    yy = "xi_0"
    temp = pd.read_csv(f"{artifact_path}/designs/{file_prefix}_designs_hist.csv",)
    plot = sns.lineplot(x="step", y=yy, data=temp, hue="order")
    plot.figure.savefig(f"mlflow_outputs/{file_prefix}_designs_hist.png")
    plot.figure.clf()

    sir_model = mlflow.pytorch.load_model(model_location, map_location="cpu")
    sir_model.SIMDATA = simdata
    ## plot optimal designs for different thetas at the end of training
    dfs = []
    if simdata:
        for i in range(10):
            idx = i + 1
            test_theta = simdata["prior_samples"][idx].unsqueeze(0)

            obsdf = sir_model.eval(theta=test_theta, verbose=False)[0]
            obsdf["theta"] = i
            dfs.append(obsdf.drop("observations", axis=1))
        dfs = pd.concat(dfs)
        plot = sns.scatterplot(x="xi_0", y="order", data=dfs, hue="theta")
        plot.figure.savefig(f"mlflow_outputs/{file_prefix}_final_designs.png")
        plot.figure.clf()

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
        # store images
        mlflow.log_param("plot", "Done")
        mlflow.log_artifact(
            f"mlflow_outputs/{file_prefix}_designs_hist.png", artifact_path="designs"
        )
        if simdata:
            mlflow.log_artifact(
                f"mlflow_outputs/{file_prefix}_final_designs.png",
                artifact_path="designs",
            )

    return


def plot_designs_experiment(experiment_id, file_prefix, simdata):
    filter_string = "params.status='complete'"
    meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)
    # run those that haven't yet been evaluated
    meta = [m for m in meta if "plot" not in m.data.params.keys()]
    experiment_run_ids = [run.info.run_id for run in meta]
    print(experiment_run_ids)
    for i, run_id in enumerate(experiment_run_ids):
        print(f"Plotting run {i+1} out of {len(experiment_run_ids)} runs...")
        plot_designs_run(
            experiment_id=experiment_id,
            run_id=run_id,
            file_prefix=file_prefix,
            simdata=simdata,
        )
        print("\n")


if __name__ == "__main__":
    ## load data for plotting
    # SIMDATA = torch.load("data/sir_sde_data.pt", map_location="cuda")
    SIMDATA = torch.load("data/sir_sde_data_small.pt", map_location="cpu")
    SIMDATA["ys"] = SIMDATA["ys"][..., 1]
    parser = argparse.ArgumentParser(description="Deep Adaptive Design: SIR plots")
    parser.add_argument("--experiment-id", default="47", type=str)
    parser.add_argument("--run-id", default="15", type=str)
    parser.add_argument("--file-prefix", default="epidemic", type=str)

    args = parser.parse_args()
    plot_designs_experiment(args.experiment_id, args.file_prefix, simdata=SIMDATA)
