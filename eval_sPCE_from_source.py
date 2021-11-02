import os
import math
import argparse
import pickle
from collections import defaultdict

import pandas as pd

import torch
import pyro

import mlflow

from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from estimators.mi import PriorContrastiveEstimation, NestedMonteCarloEstimation
from location_finding import HiddenObjects
from pharmacokinetic import Pharmacokinetic
from neural.modules import LazyFn


def make_data_source(fname, T, device="cuda", n=1):
    with open(fname, "rb") as f:
        data = pickle.load(f)

    sample = defaultdict(list)
    latent_name = "log_theta" if "pharmaco" in fname else "theta"
    for history in data["loop"]:
        sample[latent_name].append(history["theta"])

        for i in range(T):
            sample[f"y{i+1}"].append(history[f"y{i+1}"])
            sample[f"xi{i+1}"].append(history[f"xi{i+1}"])

        if len(sample[latent_name]) == n:
            record = {k: torch.cat(v).to(device) for k, v in sample.items()}
            yield record
            sample = defaultdict(list)


def eval_from_source(
    path_to_artifact, num_experiments_to_perform, num_inner_samples, seed, device,
):
    n = 1
    seed = auto_seed(seed)
    with open(path_to_artifact, "rb") as f:
        data = pickle.load(f)
    meta = data["meta"]

    if meta["model"] == "location_finding" or "locfin" in path_to_artifact:
        K, p = meta["K"], meta["p"]
        design_dim = (1, p)
        model_instance = HiddenObjects(
            design_net=LazyFn(
                lambda *args: None, prototype=torch.ones(design_dim, device=device),
            ),
            theta_loc=torch.zeros((K, p), device=device),
            theta_covmat=torch.eye(p, device=device),
            noise_scale=meta["noise_scale"] * torch.ones(1, device=device),
            p=p,
            K=K,
            T=num_experiments_to_perform[0],
        )

    elif meta["model"] == "pharmacokinetic" or "pharmaco" in path_to_artifact:
        design_dim = 1
        model_instance = Pharmacokinetic(
            design_net=LazyFn(
                lambda *args: None, prototype=torch.ones(design_dim, device=device),
            ),
            T=num_experiments_to_perform[0],
            theta_loc=torch.tensor([1, 0.1, 20], device=device).log(),
            theta_covmat=torch.eye(3, device=device) * 0.05,
        )
    else:
        raise ValueError("Unknown model.")

    EIGs_mean = pd.DataFrame(columns=["lower", "upper"])
    EIGs_se = pd.DataFrame(columns=["lower", "upper"])

    for t_exp in num_experiments_to_perform:
        data_source = make_data_source(
            fname=path_to_artifact, T=t_exp, device=device, n=n
        )

        model_instance.T = t_exp

        loss_upper = NestedMonteCarloEstimation(
            model_instance.model, n, num_inner_samples, data_source=data_source
        )
        auto_seed(seed)
        EIG_proxy_upper = torch.tensor(
            [-loss_upper.loss() for _ in range(meta["num_histories"] // n)]
        )

        data_source = make_data_source(
            fname=path_to_artifact, T=t_exp, device=device, n=n
        )
        loss_lower = PriorContrastiveEstimation(
            model_instance.model, n, num_inner_samples, data_source=data_source
        )
        auto_seed(seed)
        EIG_proxy_lower = torch.tensor(
            [-loss_lower.loss() for _ in range(meta["num_histories"] // n)]
        )

        EIGs_mean.loc[t_exp, "lower"] = EIG_proxy_lower.mean().item()
        EIGs_mean.loc[t_exp, "upper"] = EIG_proxy_upper.mean().item()
        EIGs_se.loc[t_exp, "lower"] = EIG_proxy_lower.std().item() / math.sqrt(
            meta["num_histories"] // n
        )
        EIGs_se.loc[t_exp, "upper"] = EIG_proxy_upper.std().item() / math.sqrt(
            meta["num_histories"] // n
        )

    print("EIG mean\n", EIGs_mean)
    print("EIG se\n", EIGs_se)

    return EIGs_mean, EIGs_se


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path-to-artifact", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-inner-samples", default=int(5e5), type=int)
    parser.add_argument("--num-experiments-to-perform", nargs="+", default=[10])
    args = parser.parse_args()
    args.num_experiments_to_perform = [
        int(x) if x else x for x in args.num_experiments_to_perform
    ]

    eval_from_source(
        path_to_artifact=args.path_to_artifact,
        num_experiments_to_perform=args.num_experiments_to_perform,
        num_inner_samples=args.num_inner_samples,
        seed=args.seed,
        device=args.device,
    )
