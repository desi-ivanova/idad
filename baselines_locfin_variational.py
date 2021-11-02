import os
import pickle
import argparse

import torch
from torch import nn
from pyro.infer.util import torch_item
import pyro
import pyro.distributions as dist
from tqdm import trange
import mlflow

from neural.modules import Mlp
from neural.critics import CriticBA
from neural.baselines import BatchDesignBaseline
from neural.aggregators import ConcatImplicitDAD

from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.bb_mi import BarberAgakov

from location_finding import HiddenObjects


def optimise_design_and_critic(
    posterior_loc,
    posterior_scale,
    experiment_number,
    noise_scale,
    p,
    num_sources,
    device,
    batch_size,
    num_steps,
    lr,
    annealing_scheme=None,
):
    design_init = (
        torch.distributions.Normal(0.0, 0.01)
        if experiment_number == 0
        else torch.distributions.Normal(0.0, 1.0)
    )
    design_net = BatchDesignBaseline(
        T=1, design_dim=(1, p), design_init=design_init
    ).to(device)
    new_mean = posterior_loc.reshape(num_sources, p)
    new_covmat = torch.cat(
        [
            torch.diag(x).unsqueeze(0)
            for x in (posterior_scale ** 2).reshape(num_sources, p)
        ]
    )
    ho_model = HiddenObjects(
        design_net=design_net,
        # Normal family -- new prior is stil MVN but with different params
        theta_loc=new_mean,
        theta_covmat=new_covmat,
        T=1,
        p=p,
        K=num_sources,
        noise_scale=noise_scale * torch.ones(1, device=device),
    )

    ### Set up model networks ###
    n = 1  # batch dim
    design_dim = (n, p)
    latent_dim = (num_sources, p)
    observation_dim = n
    hidden_dim = 512
    encoding_dim = 8
    hist_encoder_HD = [64, hidden_dim]
    hist_enc_critic_head_HD = [
        hidden_dim // 2,
        hidden_dim,
    ]
    ###### CRITIC NETWORKS #######
    ## history encoder
    critic_pre_pool_history_encoder = Mlp(
        input_dim=[*design_dim, observation_dim],
        hidden_dim=hist_encoder_HD,
        output_dim=encoding_dim,
    )
    critic_history_enc_head = Mlp(
        input_dim=encoding_dim,
        hidden_dim=hist_enc_critic_head_HD,
        output_dim=encoding_dim,
    )
    critic_history_encoder = ConcatImplicitDAD(
        encoder_network=critic_pre_pool_history_encoder,
        emission_network=critic_history_enc_head,
        T=1,
        empty_value=torch.ones(design_dim).to(device),
    )
    critic_net = CriticBA(
        history_encoder_network=critic_history_encoder, latent_dim=latent_dim
    ).to(device)
    ### Set-up loss ###
    mi_loss_instance = BarberAgakov(
        model=ho_model.model,
        critic=critic_net,
        batch_size=batch_size,
        prior_entropy=ho_model.theta_prior.entropy(),
    )

    ### Set-up optimiser ###
    optimizer = torch.optim.Adam
    # Annealed LR. Set gamma=1 if no annealing required
    annealing_freq, patience, factor = annealing_scheme
    scheduler = pyro.optim.ReduceLROnPlateau(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": lr},
            "factor": factor,
            "patience": patience,
            "verbose": False,
        }
    )
    oed = OED(optim=scheduler, loss=mi_loss_instance)
    ### Optimise ###
    loss_history = []
    num_steps_range = trange(0, num_steps + 0, desc="Loss: 0.000 ")
    for i in num_steps_range:
        loss = oed.step()
        # Log every 100 losses -> too slow (and unnecessary to log everything)
        if i % 100 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(loss))
            loss_eval = oed.evaluate_loss()
            # mlflow.log_metric(f"loss_{experiment_number}", loss_eval, step=i)

        # Check if lr should be decreased every 200 steps.
        # patience=5 so annealing occurs at most every 1.2K steps
        if i % annealing_freq == 0:
            scheduler.step(loss_eval)
            # store design paths

    return ho_model, critic_net


def main_loop(
    run,  # number of rollouts
    mlflow_run_id,
    device,
    T,
    noise_scale,
    num_sources,
    p,
    batch_size,
    num_steps,
    lr,
    annealing_scheme,
):
    pyro.clear_param_store()

    theta_loc = torch.zeros((num_sources, p), device=device)
    theta_covmat = torch.eye(p, device=device)
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    # sample true param
    true_theta = prior.sample(torch.Size([1]))

    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc.reshape(-1)  # check if needs to be reshaped.
    posterior_scale = torch.ones(p * num_sources, device=device)

    for t in range(0, T):
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        ho_model, critic = optimise_design_and_critic(
            posterior_loc,
            posterior_scale,
            experiment_number=t,
            noise_scale=noise_scale,
            p=p,
            num_sources=num_sources,
            device=device,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            annealing_scheme=annealing_scheme,
        )
        design, observation = ho_model.forward(theta=true_theta)
        posterior_loc, posterior_scale = critic.get_variational_params(
            *zip(design, observation)
        )
        posterior_loc, posterior_scale = (
            posterior_loc.detach(),
            posterior_scale.detach(),
        )
        designs_so_far.append(design[0])
        observations_so_far.append(observation[0])

    print(f"Fitted posterior: mean = {posterior_loc}, sd = {posterior_scale}")
    print("True theta = ", true_theta.reshape(-1))

    data_dict = {}
    for i, xi in enumerate(designs_so_far):
        data_dict[f"xi{i + 1}"] = xi.cpu()
    for i, y in enumerate(observations_so_far):
        data_dict[f"y{i + 1}"] = y.cpu()
    data_dict["theta"] = true_theta.cpu()

    return data_dict


def main(
    seed,
    mlflow_experiment_name,
    num_histories,
    device,
    T,
    p,
    num_sources,
    noise_scale,
    batch_size,
    num_steps,
    lr,
):
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)
    # Log everything
    mlflow.log_param("seed", seed)
    mlflow.log_param("p", p)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("lr", lr)
    mlflow.log_param("num_histories", num_histories)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("noise_scale", noise_scale)
    mlflow.log_param("num_sources", num_sources)
    annealing_scheme = [100, 5, 0.8]
    mlflow.log_param("annealing_scheme", str(annealing_scheme))

    meta = {
        "model": "location_finding",
        "p": p,
        "K": num_sources,
        "noise_scale": noise_scale,
        "num_histories": num_histories,
    }
    results_vi = {"loop": [], "seed": seed, "meta": meta}
    for i in range(num_histories):
        results = main_loop(
            run=i,
            mlflow_run_id=mlflow.active_run().info.run_id,
            device=device,
            T=T,
            noise_scale=noise_scale,
            num_sources=num_sources,
            p=p,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            annealing_scheme=annealing_scheme,
        )
        results_vi["loop"].append(results)

    # Log the results dict as an artifact
    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")
    with open("./mlflow_outputs/results_locfin_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)
    mlflow.log_artifact("mlflow_outputs/results_locfin_vi.pickle")
    print("Done.")
    ml_info = mlflow.active_run().info
    path_to_artifact = "mlruns/{}/{}/artifacts/results_locfin_vi.pickle".format(
        ml_info.experiment_id, ml_info.run_id
    )
    print("Path to artifact - use this when evaluating:\n", path_to_artifact)
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VI baseline: Location finding with BA bound"
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--physical-dim", default=2, type=int)
    parser.add_argument(
        "--num-histories", help="Number of histories/rollouts", default=128, type=int
    )
    parser.add_argument("--num-experiments", default=10, type=int)  # == T
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--mlflow-experiment-name", default="locfin_variational", type=str
    )
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--num-steps", default=5000, type=int)

    args = parser.parse_args()

    main(
        seed=args.seed,
        mlflow_experiment_name=args.mlflow_experiment_name,
        num_histories=args.num_histories,
        device=args.device,
        T=args.num_experiments,
        p=args.physical_dim,
        num_sources=2,
        noise_scale=0.5,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
    )
