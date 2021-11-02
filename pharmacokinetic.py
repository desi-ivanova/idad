import os
import pickle
import argparse
import math

import pandas as pd
from tqdm import trange
import torch
import torch.nn as nn
from pyro.infer.util import torch_item
import pyro
import pyro.distributions as dist

import mlflow
import mlflow.pytorch

from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.bb_mi import InfoNCE, NWJ, BarberAgakov
from estimators.mi import PriorContrastiveEstimation

from neural.modules import Mlp, SelfAttention
from neural.aggregators import PermutationInvariantImplicitDAD, ConcatImplicitDAD
from neural.baselines import BatchDesignBaseline, RandomDesignBaseline
from neural.critics import CriticDotProd, CriticBA

mi_estimator_options = {
    "NWJ": NWJ,
    "InfoNCE": InfoNCE,
    "sPCE": PriorContrastiveEstimation,
    "BA": BarberAgakov,
}


class Pharmacokinetic(nn.Module):
    """
    Pharmacokinetic model
    """

    def __init__(self, design_net, T, theta_loc=None, theta_covmat=None):
        super().__init__()
        self.p = 3  # dim of latent
        self.theta_loc = (
            theta_loc if theta_loc is not None else torch.tensor([1, 0.1, 20]).log()
        )
        self.theta_covmat = (
            theta_covmat if theta_covmat is not None else torch.eye(self.p) * 0.05
        )
        self.log_theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        )
        self.design_net = design_net
        self.T = T  # number of experiments

    ### Explicit model simulator: can be used with DAD ###
    def _simulator(
        self,
        xi,
        theta,
        D_v=400.0,
        epsilon_scale=math.sqrt(0.01),
        nu_scale=math.sqrt(0.1),
    ):
        # unpack latents [these are exp-ed already!]
        k_a, k_e, V = [theta[..., [i]] for i in range(self.p)]
        assert (k_a > k_e).all()
        # compute concentration at time t=xi
        # shape of mean is [batch, n] where n is number of obs per design
        mean = (
            (D_v / V)
            * (k_a / (k_a - k_e))
            * (
                torch.exp(-torch.einsum("...ijk, ...ik->...ij", xi, k_e))
                - torch.exp(-torch.einsum("...ijk, ...ik->...ij", xi, k_a))
            )
        )

        sd = torch.sqrt((mean * epsilon_scale) ** 2 + nu_scale ** 2)
        return dist.Normal(mean, sd).to_event(1)

    def _transform_design(self, xi_untransformed):
        return nn.Sigmoid()(xi_untransformed) * 24.0

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        ################################################################################
        # Sample theta
        ################################################################################
        # sample log-theta and exponentiat
        theta = latent_sample("log_theta", self.log_theta_prior).exp()

        y_outcomes = []
        xi_designs = []
        for t in range(self.T):
            ####################################################################
            # Get a design xi
            ####################################################################
            xi_untransformed = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            # Actual design should be between 0 and 24; we will work with
            # untransformed desgins (to avoid backpropping though sigmoids etc)
            xi = self._transform_design(xi_untransformed)

            ####################################################################
            # Sample y
            ####################################################################
            _sim = self._simulator(xi=xi, theta=theta)
            y = observation_sample(f"y{t + 1}", _sim)
            # y = observation_sample(f"y{t + 1}", self.simulator, xi, theta)

            ####################################################################
            # Update history
            ####################################################################
            y_outcomes.append(y)
            xi_designs.append(xi_untransformed)  #! work with untransformed designs

        # T-steps experiment
        return xi_designs, y_outcomes, theta

    def forward(self, log_theta):
        """Run the policy"""
        self.design_net.eval()

        def model():
            with pyro.plate_stack("expand_theta_test", [log_theta.shape[0]]):
                # condition on theta
                return pyro.condition(self.model, data={"log_theta": log_theta})()

        with torch.no_grad():
            designs, outcomes, theta = model()

        self.design_net.train()
        return designs, outcomes

    def eval(self, n_trace=3, log_theta=None, verbose=True):
        self.design_net.eval()

        if log_theta is None:
            log_theta = self.log_theta_prior.sample(torch.Size([n_trace]))
        else:
            log_theta = log_theta.unsqueeze(0).expand(n_trace, *log_theta.shape)
            # dims: [n_trace * number of thetas given, shape of theta]
            log_theta = log_theta.reshape(-1, *log_theta.shape[2:],)

        output = []
        designs, outcomes = self.forward(log_theta)
        theta = log_theta.exp()

        for i in range(n_trace):
            run_xis = []
            run_ys = []

            if verbose:
                print("Example run {}".format(i))
                print(f"*True Theta: {theta[i].cpu()}*")

            for t in range(self.T):
                xi_untransformed = designs[t][i].detach()
                xi = self._transform_design(xi_untransformed).cpu().reshape(-1)
                run_xis.append(xi)

                y = outcomes[t][i].detach().cpu().reshape(-1)
                run_ys.append(y)
                if verbose:
                    print(f"xi{t + 1}: {run_xis[-1]},  y{t + 1}: {run_ys[-1]}")

            run_df = pd.DataFrame(torch.stack(run_xis).numpy())
            run_df.columns = [f"xi_{i}" for i in range(xi.shape[0])]
            run_df["observations"] = run_ys
            run_df["order"] = list(range(1, self.T + 1))
            run_df["run_id"] = i + 1
            output.append(run_df)

        self.design_net.train()
        return pd.concat(output), theta.cpu().numpy()


def train_model(
    num_steps,
    batch_size,
    num_negative_samples,
    seed,
    lr,
    gamma,
    device,
    T,
    hidden_dim,
    encoding_dim,
    reuse_history_encoder,
    critic_arch,
    design_arch,
    mi_estimator,
    mlflow_experiment_name,
):
    pyro.clear_param_store()

    ### Set up Mlflow logging ### ------------------------------------------------------
    mlflow.set_experiment(mlflow_experiment_name)
    seed = auto_seed(seed)

    n = 1  # output dim/number of samples per design
    design_dim = (n, 1)  # design is t (time)
    latent_dim = 3
    observation_dim = n

    #### Set up and store network layers ######
    # history encoder hidden dimensions: apply to both critic and design net
    # each (xi, y) gets one encoding
    hist_encoder_HD = [64, hidden_dim]
    # design emitter hidden dimensions
    des_emitter_HD = [hidden_dim // 2, encoding_dim]

    # head layers for the CRITIC ONLY
    hist_enc_critic_head_HD = [
        hidden_dim * max(int(math.log(T)), 1),
        hidden_dim * max(int(math.log(T)), 1) // 2,
        hidden_dim,
    ]
    # latent encoder hidden dimensions: for CRITIC ONLY
    latent_encoder_HD = [8, 64, hidden_dim]

    if mi_estimator == "sPCE":
        latent_encoder_HD = []
        hist_enc_critic_head_HD = []
        critic_arch = None

    if mi_estimator == "BA":
        latent_encoder_HD = []

    if design_arch == "static":
        des_emitter_HD = []

    mlflow.log_param("HD_hist_encoder", str(hist_encoder_HD))
    mlflow.log_param("HD_des_emitter", str(des_emitter_HD))
    mlflow.log_param("HD_latent_encoder", str(latent_encoder_HD))
    mlflow.log_param("HD_hist_enc_critic_head", str(hist_enc_critic_head_HD))

    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("critic_arch", critic_arch)
    mlflow.log_param("design_arch", design_arch)
    mlflow.log_param("mi_estimator", mi_estimator)
    mlflow.log_param("reuse_history_encoder", reuse_history_encoder)
    # ----------------------------------------------------------------------------------

    ###################################################################################
    ### Setup design and critic networks
    ###################################################################################
    ### DESIGN NETWORK ###
    if design_arch == "static":
        # batch design baseline
        design_net = BatchDesignBaseline(
            T=T,
            design_dim=design_dim,
            design_init=torch.distributions.Uniform(-5.0, 5.0),
        ).to(device)
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
            name="policy_emitter",
        )

        design_net = PermutationInvariantImplicitDAD(
            history_encoder,
            design_emitter,
            empty_value=torch.ones(design_dim).to(device) * 2,
            self_attention_layer=SelfAttention(encoding_dim, encoding_dim)
            if design_arch == "attention"
            else None,
        ).to(device)

    ## can be used to e.g. pretrain critic with random designs
    design_net_init = RandomDesignBaseline(
        design_dim=design_dim,
        random_designs_dist=torch.distributions.Uniform(
            torch.tensor(-5.0, device=device), torch.tensor(5.0, device=device)
        ),
    ).to(device)

    if critic_arch is not None:
        ### CRITIC NETWORK ###
        critic_latent_encoder = Mlp(
            input_dim=latent_dim,
            hidden_dim=latent_encoder_HD,
            output_dim=encoding_dim,
            name="critic_latent_encoder",
        )
        critic_pre_pool_history_encoder = (
            history_encoder
            if reuse_history_encoder
            else Mlp(
                input_dim=[*design_dim, observation_dim],
                hidden_dim=hist_encoder_HD,
                output_dim=encoding_dim,
                name="critic_design_obs_encoder",
            )
        )
        critic_post_pool_history_encoder = Mlp(
            input_dim=encoding_dim * T if critic_arch == "cat" else encoding_dim,
            hidden_dim=hist_enc_critic_head_HD,
            output_dim=encoding_dim,
            name="critic_head",
        )
        if critic_arch == "cat":
            ######## CAT CRITIC ###############
            critic_history_encoder = ConcatImplicitDAD(
                encoder_network=critic_pre_pool_history_encoder,
                emission_network=critic_post_pool_history_encoder,
                T=T,
                empty_value=torch.ones(design_dim).to(device),
            )
        else:
            ######## SUM/ATTENTION CRITIC ###############
            critic_history_encoder = PermutationInvariantImplicitDAD(
                encoder_network=critic_pre_pool_history_encoder,
                # Critic head layers
                emission_network=critic_post_pool_history_encoder,
                empty_value=torch.ones(design_dim).to(device),
                self_attention_layer=SelfAttention(encoding_dim, encoding_dim),
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
    ###################################################################################

    ### Prior hyperparams ###
    theta_prior_loc = torch.tensor([1, 0.1, 20], device=device).log()
    # covariance of the prior
    theta_prior_covmat = torch.eye(latent_dim, device=device) * 0.05

    pharmaco = Pharmacokinetic(
        design_net=design_net_init,
        T=T,
        theta_loc=theta_prior_loc,
        theta_covmat=theta_prior_covmat,
    )

    optimizer = torch.optim.Adam
    patience = 5
    annealing_freq = 400
    mlflow.log_param("annealing_scheme", [annealing_freq, patience, gamma])

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

    if mi_estimator == "sPCE":
        mi_loss_instance = PriorContrastiveEstimation(
            model=pharmaco.model,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
        )
        pharmaco.design_net = design_net  # no critic to be trained
    elif mi_estimator == "BA":
        mi_loss_instance = BarberAgakov(
            model=pharmaco.model,
            critic=critic_net,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
            prior_entropy=pharmaco.log_theta_prior.entropy(),
        )
    else:
        mi_loss_instance = mi_estimator_options[mi_estimator](
            model=pharmaco.model,
            critic=critic_net,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
        )

    mlflow.log_param("num_negative_samples", mi_loss_instance.num_negative_samples)
    mlflow.log_param("num_batch_samples", mi_loss_instance.batch_size)

    oed = OED(optim=scheduler, loss=mi_loss_instance)

    loss_history = []
    outputs_history = []

    num_steps_range = trange(0, num_steps + 0, desc="Loss: 0.000 ")
    # eval model & store designs for this latent:
    test_log_theta = torch.tensor([1.5, 0.15, 15], device=device).log().unsqueeze(0)
    for i in num_steps_range:
        pharmaco.train()
        loss = oed.step()
        loss = torch_item(loss)
        loss_history.append(loss)
        num_steps_range.set_description("Loss: {:.3f} ".format(loss))
        # log loss every 200 steps
        if i % 200 == 0:
            loss_eval = oed.evaluate_loss()
            mlflow.log_metric("loss", loss_eval, step=i)

        # Check if lr should be decreased every <epoch_size> steps.
        if i % annealing_freq == 0:
            scheduler.step(loss_eval)
            df, latents = pharmaco.eval(
                n_trace=1, log_theta=test_log_theta, verbose=False
            )
            df["step"] = i
            outputs_history.append(df)

        if i == num_steps * 0.05:
            # switch to design network after a little bit of training of the critic only
            pharmaco.design_net = design_net

    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")

    pd.concat(outputs_history).reset_index().to_csv(f"mlflow_outputs/designs_hist.csv")
    mlflow.log_artifact(f"mlflow_outputs/designs_hist.csv", artifact_path="designs")

    pharmaco.eval()
    # store params, metrics and artifacts to mlflow ------------------------------------
    print("Storing model to MlFlow... ", end="")
    mlflow.pytorch.log_model(pharmaco.cpu(), "model")
    if critic_arch is not None:
        print("Storing critic network to MlFlow... ", end="")
        mlflow.pytorch.log_model(critic_net.cpu(), "critic")

    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts"
    print(f"Model and critic sotred in {model_loc}. Done.")
    mlflow.log_param("status", "complete")

    return pharmaco


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iDAD: Pharmacokinetic Model")
    parser.add_argument("--num-steps", default=100000, type=int)
    parser.add_argument("--num-batch-samples", default=1024, type=int)
    parser.add_argument("--num-negative-samples", default=1023, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--gamma", default=0.80, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num-experiments", default=5, type=int)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--encoding-dim", default=32, type=int)
    parser.add_argument("--reuse-history-encoder", default=False, action="store_true")
    parser.add_argument(
        "--design-arch",
        default="sum",
        help="Architecture of the design network",
        choices=["sum", "attention", "static"],
        type=str,
    )
    parser.add_argument(
        "--critic-arch",
        default="sum",
        choices=["sum", "attention", "cat"],
        help="Architecture of the critic network",
        type=str,
    )
    parser.add_argument(
        "--mi-estimator",
        default="sPCE",
        help="Mutual information estimator",
        choices=["InfoNCE", "NWJ", "sPCE", "BA"],
        type=str,
    )
    parser.add_argument("--mlflow-experiment-name", default="pharmaco", type=str)
    args = parser.parse_args()

    train_model(
        num_steps=args.num_steps,
        batch_size=args.num_batch_samples,
        num_negative_samples=args.num_negative_samples,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
        T=args.num_experiments,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        reuse_history_encoder=args.reuse_history_encoder,
        critic_arch=args.critic_arch,
        design_arch=args.design_arch,
        mi_estimator=args.mi_estimator,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
