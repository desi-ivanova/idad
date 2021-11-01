import os
import math
import argparse
from tqdm import tqdm

import pandas as pd

import torch
import pyro

import mlflow

from oed.design import OED
from estimators.bb_mi import InfoNCE, NWJ

from experiment_tools.output_utils import get_mlflow_meta


def evaluate(
    experiment_id,
    run_id,
    n_rollout,
    num_negative_samples,
    device,
    simdata,
    mi_estimator,
):
    artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"

    factor = 16
    n_rollout = n_rollout // factor

    own_critic_location = f"{artifact_path}/critic"
    mi_estimator_options = {"NWJ": NWJ, "InfoNCE": InfoNCE}

    with torch.no_grad():
        # load model and critic
        sir_model = mlflow.pytorch.load_model(model_location, map_location=device)
        sir_model.SIMDATA = simdata
        critic_net_own = mlflow.pytorch.load_model(
            own_critic_location, map_location=device
        )

        mi_own = mi_estimator_options[mi_estimator](
            model=sir_model.model,
            critic=critic_net_own,
            batch_size=factor,
            num_negative_samples=num_negative_samples,
        )

        # compute loss several times
        eig_own = []
        eig_random = []
        eig_own = torch.tensor([-mi_own.loss() for _ in range(n_rollout)])

        # compute means and std
        eig_own_mean = eig_own.mean().item()
        eig_own_std = eig_own.std().item() / math.sqrt(n_rollout)
        # eig_random_mean = eig_random.mean()

        res = pd.DataFrame(
            {"mean": eig_own_mean, "se": eig_own_std, "bound": "lower"},
            index=[sir_model.T],
        )
        res.to_csv(f"mlflow_outputs/sir_eval.csv")
        with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
            # store images
            mlflow.log_artifact(
                f"mlflow_outputs/sir_eval.csv", artifact_path="evaluation"
            )
            mlflow.log_metric("eig_own_mean", eig_own_mean)
            # mlflow.log_metric("eig_random_mean", eig_random_mean.cpu().item())

    print(eig_own_mean, eig_own_std)
    return eig_own_mean, eig_own_std


def eval_experiment(experiment_id, n_rollout, num_negative_samples, device="cuda"):
    filter_string = "params.status='complete'"
    meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)

    SIMDATA = torch.load("data/sir_sde_data_test.pt", map_location=device)
    # run those that haven't yet been evaluated
    meta = [m for m in meta if "eig_own_mean" not in m.data.metrics.keys()]

    experiment_run_ids = [run.info.run_id for run in meta]
    print(experiment_run_ids)
    for i, run_id in enumerate(experiment_run_ids):
        print(f"Evaluating run {i+1} out of {len(experiment_run_ids)} runs... {run_id}")
        evaluate(
            experiment_id=experiment_id,
            run_id=run_id,
            n_rollout=n_rollout,
            num_negative_samples=num_negative_samples,
            device=device,
            simdata=SIMDATA,
            mi_estimator=meta[i].data.params["mi_estimator"],
        )
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implicit Deep Adaptive Design: evaluate SIR model"
    )
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--n-rollout", default=2048 * 2, type=int)
    parser.add_argument("--num-negative-samples", default=10000, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    # compute validation scores
    eval_experiment(
        experiment_id=args.experiment_id,
        n_rollout=args.n_rollout,
        num_negative_samples=args.num_negative_samples,
        device=args.device,
    )
