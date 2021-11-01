from itertools import product
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pyro

import mlflow
import mlflow.pytorch
from experiment_tools.pyro_tools import auto_seed


def run_policy(implicit_model, theta=None, indices=None, verbose=True):
    """
    can specify either theta or index. If none specified it will sample.
    If both are specified theta is used and indices is ignored.
    """
    if theta is not None:
        indices = implicit_model.theta_to_index(theta)

    if indices is not None:
        # condition on thetas
        def model():
            with pyro.plate_stack("expand_theta_test", [indices.shape[0]]):
                return pyro.condition(implicit_model.model, data={"indices": indices})()

    else:
        model = implicit_model.model

    with torch.no_grad():
        trace = pyro.poutine.trace(model).get_trace()
        designs = [
            node["value"].detach()
            for node in trace.nodes.values()
            if node.get("subtype") == "design_sample"
        ]
        observations = [
            node["value"].detach()
            for node in trace.nodes.values()
            if node.get("subtype") == "observation_sample"
        ]
        latents = [
            node["value"].detach()
            for node in trace.nodes.values()
            if node.get("subtype") == "latent_sample"
        ]
        latents = torch.cat(latents, axis=-1)

    return designs, observations, latents


def plot_posterior_grid(T0, T1, pdf_post, true_theta, limits):
    vmin = 0
    vmax = np.max(pdf_post)
    levels = np.linspace(vmin, vmax, 6)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    CS_post = ax.contour(
        T0, T1, pdf_post, cmap="viridis", linewidths=1.5, levels=levels[:], zorder=10
    )

    ax.scatter(
        true_theta[0],
        true_theta[1],
        c="r",
        marker="x",
        s=200,
        zorder=20,
        label="Ground truth",
    )
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.legend(loc="upper left")
    ax.set_xlabel(r"Infection Rate $\beta$", size=20)
    ax.set_ylabel(r"Recovery Rate $\gamma$", size=20)
    ax.set_title("c) T=5 Example Posterior", size=20)
    ax.tick_params(labelsize=20)
    ax.grid(True, ls="--")

    plt.tight_layout()
    plt.show()
    # return fig, ax
    # plt.savefig(f"mlflow_outputs/SIR_posterior_example.pdf")


def get_posterior_logprob(critic, prior, mi_estimator, eval_latents, *design_obs_pairs):
    with torch.no_grad():
        foo, _ = critic(eval_latents, *design_obs_pairs)
    const = 0.0 if mi_estimator == "InfoNCE" else 1.0
    res = foo.squeeze(0) + prior.log_prob(eval_latents).sum(-1) + const
    # normalize
    return res - res.logsumexp(0)


if __name__ == "__main__":
    device = "cuda:0"
    # load data for plotting
    # NWJ: "66591b13014a4f848fb9d700d8921f83"
    # InfoNCE: f1c4ef9756b64574995d63e63359870a
    run_id = "f1c4ef9756b64574995d63e63359870a"
    with mlflow.start_run(run_id=run_id) as run:
        mi_estimator = mlflow.ActiveRun(run).data.params["mi_estimator"]
    artifact_path = f"mlruns/48/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"
    critic_location = f"{artifact_path}/critic"

    # load model and critic
    sir_model = mlflow.pytorch.load_model(model_location, map_location=device)
    critic_net = mlflow.pytorch.load_model(critic_location, map_location=device)

    scale = 0.5
    mean = torch.tensor([0.5, 0.1]).log().to(device)
    prior = torch.distributions.LogNormal(mean, scale)

    # prepare theta grid

    N_GRID = 300
    auto_seed(420)
    random_samples = prior.sample(torch.Size([50000]))
    SIMDATA = torch.load("data/sir_sde_data_test.pt", map_location=device)
    sir_model.SIMDATA = SIMDATA
    del SIMDATA
    lower, upper = random_samples.quantile(
        torch.tensor([0.01, 0.99]).to(device), axis=0
    ).cpu()
    beta_lims = [lower[0], upper[0]]
    gamma_lims = [lower[1], upper[1]]
    t_beta = torch.linspace(*beta_lims, N_GRID).to(device)
    t_gamma = torch.linspace(*gamma_lims, N_GRID).to(device)
    T0, T1 = torch.meshgrid(t_beta, t_gamma)
    theta_grid = torch.tensor(list(product(t_beta, t_gamma))).to(device)

    true_thetas = torch.tensor(
        [[0.1977, 0.1521], [0.3332, 0.1103], [0.7399, 0.0924]]
    ).to(device)
    for i, tt in enumerate(true_thetas):
        true_theta = tt.unsqueeze(0)
        nearest_theta_idx = sir_model.theta_to_index(true_theta)
        true_theta = sir_model.SIMDATA["prior_samples"][nearest_theta_idx]
        print(true_theta)
        designs, observations, latents = run_policy(sir_model, true_theta)

        foo = get_posterior_logprob(
            critic_net,
            prior,
            mi_estimator,
            theta_grid,
            *zip(list(designs), list(observations)),
        )
        plot_posterior_grid(
            T0.cpu().numpy(),
            T1.cpu().numpy(),
            foo.reshape(N_GRID, N_GRID).exp().cpu().numpy(),
            true_theta=true_theta.squeeze(0).cpu().numpy(),
            limits=[beta_lims, gamma_lims],
        )
