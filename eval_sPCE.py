import os
import math
import argparse
from collections import defaultdict

import pandas as pd

import torch
import pyro

import mlflow

from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from estimators.mi import PriorContrastiveEstimation, NestedMonteCarloEstimation


def evaluate_run(
    experiment_id,
    run_id,
    num_experiments_to_perform,
    num_inner_samples,
    device,
    n_rollout,
    seed=-1,
    # if checkpoints were stored (as model_postfix), pass here
    model_postfix="",
):
    pyro.clear_param_store()
    artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model{model_postfix}"
    seed = auto_seed(seed)

    factor = 16
    n_rollout = n_rollout // factor

    EIGs_mean = pd.DataFrame(columns=["lower", "upper"])
    EIGs_se = pd.DataFrame(columns=["lower", "upper"])

    for t_exp in num_experiments_to_perform:
        # load model, set number of experiments
        trained_model = mlflow.pytorch.load_model(model_location, map_location=device)
        if t_exp:
            trained_model.T = t_exp
        else:
            t_exp = trained_model.T

        pce_loss_upper = NestedMonteCarloEstimation(
            trained_model.model, factor, num_inner_samples
        )
        pce_loss_lower = PriorContrastiveEstimation(
            trained_model.model, factor, num_inner_samples
        )

        auto_seed(seed)
        EIG_proxy_upper = torch.tensor(
            [-pce_loss_upper.loss() for _ in range(n_rollout)]
        )
        auto_seed(seed)
        EIG_proxy_lower = torch.tensor(
            [-pce_loss_lower.loss() for _ in range(n_rollout)]
        )

        EIGs_mean.loc[t_exp, "lower"] = EIG_proxy_lower.mean().item()
        EIGs_mean.loc[t_exp, "upper"] = EIG_proxy_upper.mean().item()
        EIGs_se.loc[t_exp, "lower"] = EIG_proxy_lower.std().item() / math.sqrt(
            n_rollout
        )
        EIGs_se.loc[t_exp, "upper"] = EIG_proxy_upper.std().item() / math.sqrt(
            n_rollout
        )

    EIGs_mean["stat"] = "mean"
    EIGs_se["stat"] = "se"
    res = pd.concat([EIGs_mean, EIGs_se])
    print(res)
    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")
    res.to_csv(f"mlflow_outputs/eval{model_postfix}.csv")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
        mlflow.log_artifact(
            f"mlflow_outputs/eval{model_postfix}.csv", artifact_path="evaluation",
        )
        if len(num_experiments_to_perform) == 1:
            mlflow.log_metric(
                f"eval_mi_lower{model_postfix}", EIGs_mean.loc[t_exp, "lower"],
            )

    return res


def evaluate_experiment(
    experiment_id,
    num_experiments_to_perform=[None],
    num_inner_samples=int(5e5),
    device="cuda",
    n_rollout=2048 * 2,
    seed=-1,
    model_postfix="",
):

    filter_string = "params.status='complete'"
    meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)
    # run those that haven't yet been evaluated
    meta = [
        m for m in meta if f"eval_mi_lower{model_postfix}" not in m.data.metrics.keys()
    ]
    meta = [m for m in meta if "baseline_type" not in m.data.params.keys()]
    experiment_run_ids = [run.info.run_id for run in meta]
    print(experiment_run_ids)
    for i, run_id in enumerate(experiment_run_ids):
        print(f"Evaluating run {i+1} out of {len(experiment_run_ids)} runs...")
        evaluate_run(
            experiment_id=experiment_id,
            run_id=run_id,
            num_experiments_to_perform=num_experiments_to_perform,
            num_inner_samples=num_inner_samples,
            device=device,
            n_rollout=n_rollout,
            seed=-1,
            model_postfix=model_postfix,
        )
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design: Model Evaluation via sPCE."
    )
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--n-rollout", default=2048 * 2, type=int)
    parser.add_argument("--num-experiments-to-perform", nargs="+", default=[None])

    args = parser.parse_args()
    args.num_experiments_to_perform = [
        int(x) if x else x for x in args.num_experiments_to_perform
    ]
    evaluate_experiment(
        experiment_id=args.experiment_id,
        n_rollout=args.n_rollout,
        seed=args.seed,
        num_inner_samples=int(1e5),
        num_experiments_to_perform=args.num_experiments_to_perform,
        device=args.device,
    )
