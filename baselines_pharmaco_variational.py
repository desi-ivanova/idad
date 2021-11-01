import os
import pickle
import argparse
from tqdm import trange

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import mlflow

from neural.baselines import BatchDesignBaseline
from neural.critics import CriticBA
from neural.aggregators import ConcatImplicitDAD
from neural.modules import Mlp
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.bb_mi import BarberAgakov
from pharmacokinetic import Pharmacokinetic


def optimise_design_and_critic(
    posterior_loc,
    posterior_scale,
    experiment_number,
    device,
    batch_size,
    num_steps,
    lr,
    annealing_scheme,
):
    design_init = torch.distributions.Uniform(-5.0, 5.0)
    n = 1
    latent_dim = 3
    design_dim = (n, 1)
    design_net = BatchDesignBaseline(
        T=1, design_dim=design_dim, design_init=design_init
    ).to(device)
    new_mean = posterior_loc
    new_covmat = torch.diag(posterior_scale.reshape(-1) ** 2)
    pharmaco = Pharmacokinetic(
        design_net=design_net,
        # Normal family -- new prior is stil MVN but with different params
        theta_loc=new_mean,
        theta_covmat=new_covmat,
        T=1,
    )

    ### Set up model networks ###
    n = 1  # output dim/number of samples per design
    design_dim = (n, 1)  # design is t (time)
    latent_dim = 3  # theta dimension is
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
        empty_value=torch.ones(design_dim),
    )
    critic_net = CriticBA(
        history_encoder_network=critic_history_encoder, latent_dim=latent_dim
    ).to(device)
    ### Set-up loss ###
    mi_loss_instance = BarberAgakov(
        model=pharmaco.model,
        critic=critic_net,
        batch_size=batch_size,
        prior_entropy=pharmaco.log_theta_prior.entropy(),
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
        # Log loss every 200 steps
        if i % 100 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(loss))
            loss_eval = oed.evaluate_loss()
        if i % annealing_freq == 0:
            scheduler.step(loss_eval)

    return pharmaco, critic_net


def main_loop(
    run, mlflow_run_id, device, T, batch_size, num_steps, lr, annealing_scheme,
):
    pyro.clear_param_store()
    latent_dim = 3
    theta_loc = theta_prior_loc = torch.tensor([1, 0.1, 20], device=device).log()
    theta_covmat = torch.eye(latent_dim, device=device) * 0.05
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    print("Sampling true theta from prior")
    true_theta = prior.sample(torch.Size([1]))
    print("True theta (log):", true_theta)
    print("True theta (exp-ed):", true_theta.exp())

    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc
    posterior_scale = torch.sqrt(theta_covmat.diag())

    for t in range(0, T):
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        pharmaco, critic = optimise_design_and_critic(
            posterior_loc=posterior_loc,
            posterior_scale=posterior_scale,
            experiment_number=t,
            device=device,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            annealing_scheme=annealing_scheme,
        )
        design, observation = pharmaco.forward(log_theta=true_theta)
        posterior_loc, posterior_scale = critic.get_variational_params(
            *zip(design, observation)
        )
        posterior_loc, posterior_scale = (
            posterior_loc.detach(),
            posterior_scale.detach(),
        )
        print(f"design {design}, observation {observation}")
        print("Fitted posterior", posterior_loc, posterior_scale)
        print("True theta: ", true_theta.reshape(-1))
        designs_so_far.append(design[0])
        observations_so_far.append(observation[0])

    data_dict = {}
    for i, xi in enumerate(designs_so_far):
        data_dict[f"xi{i + 1}"] = xi.cpu()
    for i, y in enumerate(observations_so_far):
        data_dict[f"y{i + 1}"] = y.cpu()
    data_dict["theta"] = true_theta.cpu()

    return data_dict


def main(
    seed, mlflow_experiment_name, num_histories, device, T, batch_size, num_steps, lr,
):
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)
    # Log everything
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("lr", lr)
    mlflow.log_param("num_histories", num_histories)
    mlflow.log_param("num_experiments", T)
    annealing_scheme = [100, 5, 0.8]
    mlflow.log_param("annealing_scheme", str(annealing_scheme))

    results_vi = {
        "loop": [],
        "seed": seed,
        "meta": {"num_histories": num_histories, "model": "pharmacokinetic"},
    }
    for i in range(num_histories):
        results = main_loop(
            run=i,
            mlflow_run_id=mlflow.active_run().info.run_id,
            device=device,
            T=T,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            annealing_scheme=annealing_scheme,
        )
        results_vi["loop"].append(results)

    # Log the results dict as an artifact
    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")
    with open("./mlflow_outputs/results_pharmaco_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)
    mlflow.log_artifact("mlflow_outputs/results_pharmaco_vi.pickle")
    print("Done.")
    ml_info = mlflow.active_run().info
    path_to_artifact = "mlruns/{}/{}/artifacts/results_pharmaco_vi.pickle".format(
        ml_info.experiment_id, ml_info.run_id
    )
    print("Path to artifact - use this when evaluating:\n", path_to_artifact)
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VI baseline: Pharmacokinetic model")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument(
        "--num-histories", help="Number of histories/rollouts", default=128, type=int
    )
    parser.add_argument("--num-experiments", default=10, type=int)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--mlflow-experiment-name", default="pharmacokinetic_variational", type=str
    )
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num-steps", default=5000, type=int)

    args = parser.parse_args()

    main(
        seed=args.seed,
        num_histories=args.num_histories,
        device=args.device,
        T=args.num_experiments,
        lr=args.lr,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
