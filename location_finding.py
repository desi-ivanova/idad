import os
import pickle
import argparse

import math
import pandas as pd

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from tqdm import trange
import mlflow
import mlflow.pytorch

from neural.modules import Mlp, SelfAttention
from neural.aggregators import PermutationInvariantImplicitDAD, ConcatImplicitDAD
from neural.baselines import BatchDesignBaseline
from neural.critics import CriticDotProd, CriticBA

from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.mi import PriorContrastiveEstimation
from estimators.bb_mi import InfoNCE, NWJ, BarberAgakov


mi_estimator_options = {
    "sPCE": PriorContrastiveEstimation,
    "NWJ": NWJ,
    "InfoNCE": InfoNCE,
    "BA": BarberAgakov,
}


class HiddenObjects(nn.Module):
    """Location finding example"""

    def __init__(
        self,
        design_net,
        base_signal=0.1,  # G-map hyperparam
        max_signal=1e-4,  # G-map hyperparam
        theta_loc=None,  # prior on theta mean hyperparam
        theta_covmat=None,  # prior on theta covariance hyperparam
        noise_scale=None,  # this is the scale of the noise term
        p=1,  # physical dimension
        K=1,  # number of sources
        T=2,  # number of experiments
    ):
        super().__init__()
        self.design_net = design_net
        self.base_signal = base_signal
        self.max_signal = max_signal
        # Set prior:
        self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((K, p))
        self.theta_covmat = theta_covmat if theta_covmat is not None else torch.eye(p)
        self.theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        ).to_event(1)
        # Observations noise scale:
        self.noise_scale = noise_scale if noise_scale is not None else torch.tensor(1.0)
        self.n = 1  # samples per design=1
        self.p = p  # dimension of theta (location finding example will be 1, 2 or 3).
        self.K = K  # number of sources
        self.T = T  # number of experiments

    def forward_map(self, xi, theta):
        """Defines the forward map for the hidden object example
        y = G(xi, theta) + Noise.
        """
        # two norm squared
        sq_two_norm = (xi - theta).pow(2).sum(axis=-1)
        # add a small number before taking inverse (determines max signal)
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
        # sum over the K sources, add base signal and take log.
        mean_y = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True))
        return mean_y

    def model(self):
        if hasattr(self.design_net, "parameters"):
            #! this is required for the pyro optimizer
            pyro.module("design_net", self.design_net)

        ########################################################################
        # Sample latent variables theta
        ########################################################################
        theta = latent_sample("theta", self.theta_prior)
        y_outcomes = []
        xi_designs = []

        # T-steps experiment
        for t in range(self.T):
            ####################################################################
            # Get a design xi; shape is [batch size x self.n x self.p]
            ####################################################################
            xi = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            ####################################################################
            # Sample y at xi; shape is [batch size x 1]
            ####################################################################
            mean = self.forward_map(xi, theta)
            sd = self.noise_scale
            y = observation_sample(f"y{t + 1}", dist.Normal(mean, sd).to_event(1))

            ####################################################################
            # Update history
            ####################################################################
            y_outcomes.append(y)
            xi_designs.append(xi)

        return theta, xi_designs, y_outcomes

    def forward(self, theta):
        """Run the policy for a given theta"""
        self.design_net.eval()

        def conditioned_model():
            with pyro.plate_stack("expand_theta_test", [theta.shape[0]]):
                # condition on theta
                return pyro.condition(self.model, data={"theta": theta})()

        with torch.no_grad():
            theta, designs, observations = conditioned_model()
        self.design_net.train()
        return designs, observations

    def eval(self, n_trace=3, theta=None, verbose=True):
        """run the policy, print output and return it in a dataframe"""
        self.design_net.eval()

        if theta is None:
            theta = self.theta_prior.sample(torch.Size([n_trace]))
        else:
            theta = theta.unsqueeze(0).expand(n_trace, *theta.shape)
            # dims: [n_trace * number of thetas given, shape of theta]
            theta = theta.reshape(-1, *theta.shape[2:])

        designs, observations = self.forward(theta)
        output = []
        true_thetas = []

        for i in range(n_trace):
            if verbose:
                print("\nExample run {}".format(i + 1))
                print(f"*True Theta: {theta[i].cpu()}*")
            run_xis = []
            run_ys = []
            # Print optimal designs, observations for given theta
            for t in range(self.T):
                xi = designs[t][i].detach().cpu().reshape(-1)
                run_xis.append(xi)
                y = observations[t][i].detach().cpu().item()
                run_ys.append(y)
                if verbose:
                    print(f"xi{t + 1}: {xi},   y{t + 1}: {y}")
            run_df = pd.DataFrame(torch.stack(run_xis).numpy())
            run_df.columns = [f"xi_{i}" for i in range(self.p)]
            run_df["observations"] = run_ys
            run_df["order"] = list(range(1, self.T + 1))
            run_df["run_id"] = i + 1
            output.append(run_df)

        self.design_net.train()
        return pd.concat(output), theta.cpu().numpy()


def single_run(
    seed,
    num_steps,
    batch_size,  # N to estimate outer E
    num_negative_samples,  # L in denom
    lr,  # learning rate of adam optim
    gamma,  # scheduler for adam optim
    p,  # number of physical dim
    K,  # number of sources
    T,  # number of experiments
    noise_scale,
    device,
    hidden_dim,
    encoding_dim,
    mlflow_experiment_name,
    design_arch,  # "sum" or "attention" or "static"
    reuse_history_encoder,
    critic_arch,
    mi_estimator,
):

    pyro.clear_param_store()
    seed = auto_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)

    ### Set up model networks ###
    n = 1  # batch dim
    design_dim = (n, p)
    latent_dim = (K, p)
    observation_dim = n

    # history encoder hidden dimensions: apply to both critic and design net
    # each (xi, y) gets one encoding
    hist_encoder_HD = [64, hidden_dim]
    # design emitter hidden dimensions
    des_emitter_HD = [hidden_dim // 2, encoding_dim]

    # head layers for the CRITIC ONLY
    hist_enc_critic_head_HD = [
        hidden_dim * int(max(math.log(T), 1)),
        hidden_dim * int(max(math.log(T), 1)) // 2,
        hidden_dim,
    ]
    # latent encoder hidden dimensions: for CRITIC ONLY
    latent_encoder_HD = [16, 64, hidden_dim]

    if mi_estimator == "sPCE":
        latent_encoder_HD = []
        hist_enc_critic_head_HD = []
        critic_arch = None

    if mi_estimator == "BA":
        # BA bound doesn't have a latent encoder
        latent_encoder_HD = []

    if design_arch == "static":
        des_emitter_HD = []

    mlflow.log_param(
        "design_arch_full", f"{hist_encoder_HD}, {design_arch}, {des_emitter_HD}"
    )
    # mlflow.log_param("HD_hist_encoder", str(hist_encoder_HD))
    # mlflow.log_param("HD_des_emitter", str(des_emitter_HD))
    mlflow.log_param("critic_latent_encoder", f"{latent_encoder_HD}")
    mlflow.log_param(
        "critic_history_encoder",
        f"{hist_encoder_HD}, {critic_arch}, {hist_enc_critic_head_HD}",
    )

    if design_arch == "static":
        # batch design baseline
        design_net = BatchDesignBaseline(T=T, design_dim=design_dim,).to(device)
    else:
        history_encoder = Mlp(
            input_dim=[*design_dim, observation_dim],
            hidden_dim=hist_encoder_HD,
            output_dim=encoding_dim,
            name="policy_history_encoder",
        )
        design_emitter = Mlp(
            input_dim=encoding_dim,
            hidden_dim=des_emitter_HD,
            output_dim=design_dim,
            name="policy_design_emitter",
        )
        # Design net: takes pairs [design, observation] as input
        design_net = PermutationInvariantImplicitDAD(
            history_encoder,
            design_emitter,
            empty_value=torch.zeros(design_dim).to(device),
            self_attention_layer=SelfAttention(encoding_dim, encoding_dim)
            if design_arch == "attention"
            else None,
        ).to(device)

    if critic_arch is not None:
        ######  nets for the critic ######
        critic_latent_encoder = Mlp(
            input_dim=[*latent_dim, 0],
            hidden_dim=latent_encoder_HD,
            output_dim=encoding_dim,
            name="critic_latent_encoder",
        )
        critic_design_outcome_encoder = (
            history_encoder
            if reuse_history_encoder
            else Mlp(
                input_dim=[*design_dim, observation_dim],
                hidden_dim=hist_encoder_HD,
                output_dim=encoding_dim,
                name="critic_design_outcome_encoder",
            )
        )
        critic_head = Mlp(
            input_dim=encoding_dim * T if critic_arch == "cat" else encoding_dim,
            hidden_dim=hist_enc_critic_head_HD,
            output_dim=encoding_dim,
            name="critic_head",
        )
        if critic_arch == "cat":
            critic_history_encoder = ConcatImplicitDAD(
                encoder_network=critic_design_outcome_encoder,
                emission_network=critic_head,
                empty_value=torch.zeros(design_dim).to(device),
                T=T,
            )
        else:
            critic_history_encoder = PermutationInvariantImplicitDAD(
                encoder_network=critic_design_outcome_encoder,
                # pass head layer as an emitter
                emission_network=critic_head,
                empty_value=torch.ones(design_dim).to(device),
                self_attention_layer=SelfAttention(encoding_dim, encoding_dim)
                if design_arch == "attention"
                else None,
            )

        if mi_estimator == "BA":
            critic_net = CriticBA(
                history_encoder_network=critic_history_encoder, latent_dim=latent_dim
            ).to(device)
        else:
            critic_net = CriticDotProd(
                history_encoder_network=critic_history_encoder,
                latent_encoder_network=critic_latent_encoder,
            ).to(device)

    else:
        critic_net = None

    ### Set up Mlflow logging ### ------------------------------------------------------
    ## Reproducibility
    mlflow.log_param("seed", seed)
    ## Model hyperparams
    mlflow.log_param("noise_scale", noise_scale)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("num_sources", K)
    mlflow.log_param("physical_dim", p)

    ## Design network hyperparams
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("design_arch", design_arch)
    mlflow.log_param("reuse_history_encoder", reuse_history_encoder)
    mlflow.log_param("critic_arch", critic_arch)
    mlflow.log_param("mi_estimator", mi_estimator)

    ## Optimiser hyperparams
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("lr", lr)
    mlflow.log_param("gamma", gamma)
    # ----------------------------------------------------------------------------------

    ### Prior hyperparams ###
    # The prior is K independent * p-variate Normals. For example, if there's 1 source
    # (K=1) in 2D (p=2), then we have 1 bivariate Normal.
    theta_prior_loc = torch.zeros((K, p), device=device)  # mean of the prior
    theta_prior_covmat = torch.eye(p, device=device)  # covariance of the prior
    # noise of the model: the sigma in N(G(theta, xi), sigma)
    noise_scale_tensor = noise_scale * torch.tensor(
        1.0, dtype=torch.float32, device=device
    )

    ### Set-up optimiser ###
    optimizer = torch.optim.Adam
    patience = 5
    epoch_size = 400
    num_epochs = num_steps // epoch_size

    design_net_init = design_net

    def separate_learning_rate(module_name, param_name):
        # option to define different learning rates for design and critic
        if module_name == "critic_net":
            return {"lr": lr}
        elif module_name == "design_net":
            return {"lr": lr}
        else:
            raise NotImplementedError()

    scheduler = pyro.optim.ReduceLROnPlateau(
        {
            "optimizer": optimizer,
            "optim_args": separate_learning_rate,
            "factor": gamma,
            "patience": patience,
            "verbose": False,
        }
    )
    mlflow.log_param("annealing_scheme", [epoch_size, patience, gamma])

    ho_model = HiddenObjects(
        design_net=design_net_init,
        theta_loc=theta_prior_loc,
        theta_covmat=theta_prior_covmat,
        noise_scale=noise_scale_tensor,
        p=p,
        K=K,
        T=T,
    )

    ### Set-up loss ###
    if mi_estimator == "sPCE":
        mi_loss_instance = PriorContrastiveEstimation(
            model=ho_model.model,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
        )
    elif mi_estimator == "BA":
        mi_loss_instance = BarberAgakov(
            model=ho_model.model,
            critic=critic_net,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
            prior_entropy=ho_model.theta_prior.entropy(),
        )
    else:
        mi_loss_instance = mi_estimator_options[mi_estimator](
            model=ho_model.model,
            critic=critic_net,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
        )

    mlflow.log_param("num_negative_samples", mi_loss_instance.num_negative_samples)
    mlflow.log_param("num_batch_samples", mi_loss_instance.batch_size)

    oed = OED(optim=scheduler, loss=mi_loss_instance)

    ### Optimise ###
    loss_history = []
    outputs_history = []

    num_steps_range = trange(1, num_steps + 1, desc="Loss: 0.000 ")
    with torch.no_grad():
        test_theta = ho_model.theta_prior.sample(torch.Size([1]))

    for i in num_steps_range:
        ho_model.train()
        loss = oed.step()
        # Log every 200 losses -> too slow (and unnecessary) to log everything
        if (i - 1) % 200 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(loss))
            loss_eval = oed.evaluate_loss()
            mlflow.log_metric("loss", loss_eval, step=i)

        # Check if lr should be decreased every <epoch_size> steps.
        if i % epoch_size == 0:
            scheduler.step(loss_eval)
            # store design paths
            df, latents = ho_model.eval(n_trace=3, theta=test_theta, verbose=False)
            df["step"] = i
            outputs_history.append(df)

    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")

    if len(outputs_history) > 0:
        pd.concat(outputs_history).reset_index().to_csv(
            f"mlflow_outputs/designs_hist.csv"
        )
        mlflow.log_artifact(f"mlflow_outputs/designs_hist.csv", artifact_path="designs")

    ho_model.eval()
    # Store model and critic as artifacts
    print("Storing model to MlFlow... ", end="")
    mlflow.pytorch.log_model(ho_model.cpu(), "model")
    if critic_arch is not None:
        print("Storing critic network to MlFlow... ", end="")
        mlflow.pytorch.log_model(critic_net.cpu(), "critic")

    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts"
    print(f"Model and critic sotred in {model_loc}. Done.")

    mlflow.log_param("status", "complete")
    return ho_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iDAD: Hidden Object Detection.")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=100000, type=int)
    parser.add_argument("--num-negative-samples", default=2047, type=int)
    parser.add_argument("--num-batch-samples", default=2048, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--physical-dim", default=2, type=int)
    parser.add_argument("--num-experiments", default=10, type=int)  # == T
    parser.add_argument("--num-sources", default=2, type=int)  # == K
    parser.add_argument("--noise-scale", default=0.5, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--encoding-dim", default=64, type=int)
    parser.add_argument("--design-arch", default="attention", type=str)
    parser.add_argument("--reuse-history-encoder", default=False, action="store_true")
    parser.add_argument("--critic-arch", default="attention", type=str)
    parser.add_argument("--mi-estimator", default="InfoNCE", type=str)
    parser.add_argument("--mlflow-experiment-name", default="locfin", type=str)

    args = parser.parse_args()

    single_run(
        seed=args.seed,
        num_steps=args.num_steps,
        batch_size=args.num_batch_samples,
        num_negative_samples=args.num_negative_samples,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
        p=args.physical_dim,
        K=args.num_sources,
        T=args.num_experiments,
        noise_scale=args.noise_scale,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        mlflow_experiment_name=args.mlflow_experiment_name,
        design_arch=args.design_arch,
        reuse_history_encoder=args.reuse_history_encoder,
        critic_arch=args.critic_arch,
        mi_estimator=args.mi_estimator,
    )
