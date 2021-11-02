import os
import pickle
import argparse
from collections import OrderedDict

import torch
from torch import nn
import pyro
import pyro.distributions as dist
from tqdm import trange
import mlflow

from neural.modules import (
    LazyFn,
    BatchDesignBaseline,
    MlpEncoder,
    MlpEmitter,
    CriticBA,
    CatHistoryEncoder,
)
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.bb_mi import BarberAgakov

from epidemic import Epidemic
from epidemic_simulate_data import solve_sir_sdes


def optimise_design_and_critic(
    simdata,
    previous_design,
    posterior_loc,
    posterior_scale,
    experiment_number,
    device,
    batch_size,
    num_steps,
    lr,
    lr_critic,
    annealing_scheme=None,
):

    design_init = torch.distributions.Uniform(-3.0, 3.0)
    design_net = BatchDesignBaseline(T=1, design_dim=1, design_init=design_init).to(
        device
    )
    new_mean = posterior_loc
    new_covmat = torch.diag(posterior_scale ** 2)
    posterior_dist = torch.distributions.MultivariateNormal(new_mean, new_covmat)
    epidemic = Epidemic(
        design_net=design_net,
        T=1,
        design_transform="iid",
        simdata=simdata,
        lower_bound=previous_design.to(device),
        upper_bound=torch.tensor(100.0 - 1e-2, device=device),
    )

    ### Set up model networks ###
    n = 1  # output dim/number of samples per design
    design_dim = 1  # design is t (time)
    latent_dim = 2  #
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
    critic_pre_pool_history_encoder = MlpEncoder(
        input_dim=[design_dim, 0, observation_dim],
        hidden_dim=hist_encoder_HD,
        encoding_dim=encoding_dim,
    )
    critic_history_enc_head = MlpEncoder(
        input_dim=encoding_dim,
        hidden_dim=hist_enc_critic_head_HD,
        encoding_dim=encoding_dim,
    )
    critic_history_encoder = CatHistoryEncoder(
        encoder_network=critic_pre_pool_history_encoder,
        emission_network=critic_history_enc_head,
    )
    critic_net = CriticBA(
        history_encoder_network=critic_history_encoder,
        latent_dim=latent_dim,
        head_layer_mean=nn.Sequential(
            OrderedDict(
                [
                    ("critic_ba_l1_mean", nn.Linear(encoding_dim, 512)),
                    ("critic_ba_relu1_mean", nn.ReLU()),
                    ("critic_ba_output_mean", nn.Linear(512, 2)),
                    # mean should be positive
                    ("critic_ba_output_activation", nn.Softplus()),
                ]
            )
        ),
    ).to(device)
    ### Set-up loss ###
    mi_loss_instance = BarberAgakov(
        model=epidemic.model,
        critic=critic_net,
        batch_size=batch_size,
        # posterior is the new prior:
        prior_entropy=posterior_dist.entropy(),
    )

    ### Set-up optimiser ###
    optimizer = torch.optim.Adam
    # Annealed LR. Set gamma=1 if no annealing required
    annealing_freq, patience, factor = annealing_scheme

    def separate_learning_rate(module_name, param_name):
        if module_name == "critic_net":
            return {"lr": lr_critic}
        elif module_name == "design_net":
            return {"lr": lr}
        else:
            raise NotImplementedError()

    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": separate_learning_rate,  # {"lr": lr},
            "gamma": factor,
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
        if i % annealing_freq == 0:
            scheduler.step()

    return epidemic, critic_net


def main_loop(
    run,  # number of rollouts
    mlflow_run_id,
    device,
    T,
    batch_size,
    num_steps,
    lr,
    lr_critic,
    annealing_scheme=None,
    true_theta=None,
):
    pyro.clear_param_store()
    SIMDATA = torch.load("data/sir_sde_data.pt", map_location=device)
    latent_dim = 2
    theta_loc = theta_prior_loc = torch.tensor([0.5, 0.1], device=device).log()
    theta_covmat = torch.eye(2, device=device) * 0.5 ** 2
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    if true_theta is None:
        true_theta = prior.sample(torch.Size([1])).exp()

    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc
    posterior_scale = torch.sqrt(theta_covmat.diag())
    previous_design = torch.tensor(0.0, device=device)  # no previous design
    for t in range(0, T):
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        if t > 0:
            # update previous design so the lower bound gets updated!
            # e.forward method of Epidemic designs/obs stuff in lists
            previous_design = design_transformed[0].reshape(-1)
            ## pre-simulate data using the posterior as the prior!
            epidemic._remove_data()
            del SIMDATA
            SIMDATA = solve_sir_sdes(
                num_samples=100000,
                device=device,
                grid=10000,
                save=False,
                savegrad=False,
                theta_loc=posterior_loc.log().reshape(-1),
                theta_covmat=torch.diag(posterior_scale.reshape(-1) ** 2),
            )
            SIMDATA = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in SIMDATA.items()
            }
        epidemic, critic = optimise_design_and_critic(
            simdata=SIMDATA,
            previous_design=previous_design,
            posterior_loc=posterior_loc.reshape(-1),
            posterior_scale=posterior_scale.reshape(-1),
            experiment_number=t,
            device=device,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            lr_critic=lr_critic,
            annealing_scheme=annealing_scheme,
        )
        design_untransformed, design_transformed, observation = epidemic.forward(
            theta=true_theta
        )
        posterior_loc, posterior_scale = critic.get_variational_params(
            *zip(design_untransformed, observation)
        )
        posterior_loc, posterior_scale = (
            posterior_loc.detach(),
            posterior_scale.detach(),
        )
        designs_so_far.append(design_untransformed[0])
        observations_so_far.append(observation[0])

    print(f"Final posterior: mean = {posterior_loc}, sd = {posterior_scale}")
    print(f"True theta = {true_theta}")

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
    num_loop,
    device,
    T,
    batch_size,
    num_steps,
    lr,
    lr_critic,
):
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)
    # Log everything
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("lr", lr)
    mlflow.log_param("lr_critic", lr_critic)
    mlflow.log_param("num_loop", num_loop)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("batch_size", batch_size)
    annealing_scheme = [500, 5, 0.96]
    mlflow.log_param("annealing_scheme", str(annealing_scheme))

    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")

    results_vi = {"loop": [], "seed": seed}
    for i in range(num_loop):
        # for i, tt in enumerate(true_thetas):
        results = main_loop(
            run=i,
            mlflow_run_id=mlflow.active_run().info.run_id,
            device=device,
            T=T,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr / (i + 1),
            lr_critic=lr_critic / (i + 1),
            annealing_scheme=annealing_scheme,
        )
        results_vi["loop"].append(results)

    # Log the results dict as an artifact
    with open("./mlflow_outputs/results_sir_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)
    mlflow.log_artifact("mlflow_outputs/results_sir_vi.pickle")
    print("Done.")

    # --------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VI baseline: SIR model")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-loop", default=32, type=int)
    parser.add_argument("--batch-size", default=512, type=int)  # == T
    parser.add_argument("--num-experiments", default=5, type=int)  # == T
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--mlflow-experiment-name", default="epidemic_variational", type=str
    )
    parser.add_argument("--lr-design", default=0.01, type=float)
    parser.add_argument("--lr-critic", default=0.001, type=float)
    parser.add_argument("--num-steps", default=5000, type=int)

    args = parser.parse_args()

    main(
        seed=args.seed,
        num_loop=args.num_loop,
        device=args.device,
        batch_size=args.batch_size,
        T=args.num_experiments,
        lr=args.lr_design,
        lr_critic=args.lr_critic,
        num_steps=args.num_steps,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
