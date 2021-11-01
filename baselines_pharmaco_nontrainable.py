import os
import math
import argparse
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn
import pyro
import mlflow

from pharmacokinetic import Pharmacokinetic
from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from estimators.mi import PriorContrastiveEstimation, NestedMonteCarloEstimation
from neural.aggregators import ImplicitDeepAdaptiveDesign
from neural.baselines import RandomDesignBaseline, ConstantBatchBaseline


def evaluate_nontrainable_policy_pk(
    mlflow_experiment_name,
    num_experiments_to_perform,
    policy,  # random or equal_interval
    device,
    n_rollout=2048 * 2,
    num_inner_samples=int(5e5),
    seed=-1,
):
    """ T designs at equal intervals """
    pyro.clear_param_store()
    seed = auto_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.log_param("seed", seed)
    mlflow.log_param("baseline_type", policy)
    mlflow.log_param("n_rollout", n_rollout)
    mlflow.log_param("num_inner_samples", num_inner_samples)

    factor = 16
    n_rollout = n_rollout // factor
    n = 1
    design_dim = (n, 1)

    EIGs = pd.DataFrame(
        columns=["mean_lower", "se_lower", "mean_upper", "se_upper"],
        index=num_experiments_to_perform,
    )
    theta_prior_loc = torch.tensor([1, 0.1, 20], device=device).log()
    # covariance of the prior
    theta_prior_covmat = torch.eye(3, device=device) * 0.05

    uniform_sampler = torch.distributions.Uniform(
        torch.tensor(-5.0, device=device), torch.tensor(5.0, device=device)
    )
    for T in num_experiments_to_perform:
        if policy == "equal_interval":
            # ASSUMPTION: first design 5 min after administation
            transformed_designs = (
                torch.linspace(5.0 / 60, 23.9, T, dtype=torch.float32) / 24.0
            )
            equispaced_constant_policy = torch.log(
                transformed_designs / (1 - transformed_designs)
            ).to(device)
            design_net = ConstantBatchBaseline(
                const_designs_list=equispaced_constant_policy.unsqueeze(1),
                design_dim=design_dim,
            )
        elif policy == "random":
            design_net = RandomDesignBaseline(
                design_dim=design_dim, random_designs_dist=uniform_sampler
            )

        # Model and losses
        pk_model = Pharmacokinetic(
            design_net=design_net,
            T=T,
            theta_loc=theta_prior_loc,
            theta_covmat=theta_prior_covmat,
        )
        pce_loss_lower = PriorContrastiveEstimation(
            pk_model.model, factor, num_inner_samples
        )
        pce_loss_upper = NestedMonteCarloEstimation(
            pk_model.model, factor, num_inner_samples
        )

        auto_seed(seed)
        EIG_proxy_lower = torch.tensor(
            [-pce_loss_lower.loss() for _ in range(n_rollout)]
        )
        auto_seed(seed)
        EIG_proxy_upper = torch.tensor(
            [-pce_loss_upper.loss() for _ in range(n_rollout)]
        )

        EIGs.loc[T, "mean_lower"] = EIG_proxy_lower.mean().item()
        EIGs.loc[T, "se_lower"] = EIG_proxy_lower.std().item() / math.sqrt(n_rollout)
        EIGs.loc[T, "mean_upper"] = EIG_proxy_upper.mean().item()
        EIGs.loc[T, "se_upper"] = EIG_proxy_upper.std().item() / math.sqrt(n_rollout)

    EIGs.to_csv(f"mlflow_outputs/eval.csv")
    mlflow.log_artifact(f"mlflow_outputs/eval.csv", artifact_path="evaluation")
    mlflow.log_param("status", "complete")

    print(EIGs)
    print("Done!")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="iDAD: Pharmacokinetic model,nontrainable baselines."
    )
    parser.add_argument(
        "--mlflow-experiment-name", default="pharmaco_baselines_nontrainable", type=str,
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument(
        "--policy", default="random", choices=["random", "equal_interval"], type=str
    )
    parser.add_argument("--num-experiments-to-perform", nargs="+", default=[5, 10])
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()
    args.num_experiments_to_perform = [
        int(x) if x else x for x in args.num_experiments_to_perform
    ]
    evaluate_nontrainable_policy_pk(
        mlflow_experiment_name=args.mlflow_experiment_name,
        num_experiments_to_perform=args.num_experiments_to_perform,
        device=args.device,
        policy=args.policy,
        seed=args.seed,
    )
