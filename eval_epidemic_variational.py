import pickle
import math
import torch
import pyro

import mlflow
from mlflow.tracking import MlflowClient
from oed.design import OED


from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from estimators.bb_mi import InfoNCE, NWJ
from epidemic import Epidemic

device = "cpu"
# SIMDATA = torch.load("data/sir_sde_data_test.pt", map_location=device)

path_to_artifact = (
    "mlruns/58/d897220fdeb447dc91bb4fd04af864ad/artifacts/results_sir_vi.pickle"
)
paths_to_artifact = [
    "mlruns/58/b621c97e29674d4b86b2ec03a8581145/artifacts/results_sir_vi.pickle",
    "mlruns/58/87842d0557254f3f8ab7ddedbcead3d0/artifacts/results_sir_vi_8.pickle",
]
with open(path_to_artifact, "rb") as f:
    epidemic_vi_results = pickle.load(f)
for path in paths_to_artifact:
    with open(path_to_artifact, "rb") as f:
        temp = pickle.load(f)
    epidemic_vi_results["loop"] += temp["loop"]

# 1efceb0c3f344b1da08039f2523b43a2
random_critic_nce = mlflow.pytorch.load_model(
    "mlruns/57/464154fe078a4230adc22a9e3541ada0/artifacts/critic", map_location=device
)
random_critic_nwj = mlflow.pytorch.load_model(
    "mlruns/48/66591b13014a4f848fb9d700d8921f83/artifacts/critic", map_location=device
)

eval_latents = torch.cat(
    [sample["theta"] for sample in epidemic_vi_results["loop"]]
).to(device)
designs = []
observations = []
for i in range(1, 6):
    designs.append(
        torch.cat([sample[f"xi{i}"] for sample in epidemic_vi_results["loop"]]).to(
            device
        )
    )
    observations.append(
        torch.cat([sample[f"y{i}"] for sample in epidemic_vi_results["loop"]]).to(
            device
        )
    )

# if mi_estimator == "NWJ":
with torch.no_grad():
    joint_scores, product_scores = random_critic_nwj(
        eval_latents, *zip(designs, observations)
    )
batch_size = len(epidemic_vi_results["loop"])
num_negative_samples = batch_size - 1
joint_term = joint_scores.sum() / batch_size
product_term = (
    (product_scores.exp().sum() - batch_size)
    * math.exp(-1)
    / (batch_size * num_negative_samples)
)
MI_NWJ = joint_term - product_term
print(MI_NWJ)

with torch.no_grad():
    joint_scores, product_scores = random_critic_nce(
        eval_latents, *zip(designs, observations)
    )
joint_term = joint_scores[:batch_size].sum() / batch_size

product_term = (joint_scores + product_scores)[:batch_size].logsumexp(dim=1).mean()
MI_nce = joint_term - product_term + math.log(num_negative_samples + 1)
print(MI_nce)
