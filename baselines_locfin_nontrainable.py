import math
import argparse
from tqdm import tqdm

import pandas as pd

import torch
import torch.nn as nn
import pyro

import mlflow

from location_finding import HiddenObjects

from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from estimators.mi import PriorContrastiveEstimation, NestedMonteCarloEstimation
from neural.aggregators import ImplicitDeepAdaptiveDesign
from neural.baselines import RandomDesignBaseline


def evaluate_nontrainable_policy_locfin(
    mlflow_experiment_name,
    num_experiments_to_perform,
    device,
    policy="random",
    K=2,
    p=2,
    n_rollout=2 * 2048,
    num_inner_samples=int(5e5),
    seed=-1,
):
    """ T designs at equal intervals """
    pyro.clear_param_store()
    seed = auto_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.log_param("seed", seed)
    mlflow.log_param("K", K)
    mlflow.log_param("p", p)
    mlflow.log_param("baseline_type", "random")
    mlflow.log_param("n_rollout", n_rollout)
    mlflow.log_param("num_inner_samples", num_inner_samples)

    factor = 16
    n_rollout = n_rollout // factor
    EIGs = pd.DataFrame(
        columns=["mean_lower", "se_lower", "mean_upper", "se_upper"],
        index=num_experiments_to_perform,
    )

    theta_prior_loc = torch.zeros((K, p), device=device)
    theta_prior_covmat = torch.eye(p, device=device)
    design_dim = (1, p)
    normal_sampler = torch.distributions.Normal(
        torch.zeros(design_dim, device=device), torch.ones(design_dim, device=device)
    )

    for T in num_experiments_to_perform:
        design_net = RandomDesignBaseline(
            design_dim=design_dim, random_designs_dist=normal_sampler
        ).to(device)
        # Model and losses
        locfin_model = HiddenObjects(
            design_net=design_net,
            T=T,
            theta_loc=theta_prior_loc,
            theta_covmat=theta_prior_covmat,
            K=K,
            p=p,
            noise_scale=torch.tensor(0.5, device=device),
        )
        pce_loss_lower = PriorContrastiveEstimation(
            locfin_model.model, factor, num_inner_samples
        )
        pce_loss_upper = NestedMonteCarloEstimation(
            locfin_model.model, factor, num_inner_samples
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
        mlflow.log_param(f"eig_lower_{T}", EIG_proxy_lower.mean().item())

    EIGs.to_csv(f"mlflow_outputs/eval.csv")
    mlflow.log_artifact(f"mlflow_outputs/eval.csv", artifact_path="evaluation")
    mlflow.log_param("status", "complete")
    print(EIGs)
    print("Done!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="iDAD: Location Finding Nontrainable Baselines."
    )
    parser.add_argument(
        "--mlflow-experiment-name", default="locfin_nontrainable_baselines", type=str
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--policy", default="random", choices=["random"], type=str)
    parser.add_argument("--num-experiments-to-perform", nargs="+", default=[5, 10, 20])
    parser.add_argument("--physical-dim", default=2, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()
    args.num_experiments_to_perform = [
        int(x) if x else x for x in args.num_experiments_to_perform
    ]
    evaluate_nontrainable_policy_locfin(
        seed=args.seed,
        mlflow_experiment_name=args.mlflow_experiment_name,
        num_experiments_to_perform=args.num_experiments_to_perform,
        policy=args.policy,
        p=args.physical_dim,
        device=args.device,
    )
