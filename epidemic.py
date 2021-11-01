import os
import pickle
import argparse
import math

import pandas as pd
from tqdm import trange
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import mlflow
import mlflow.pytorch

from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.bb_mi import InfoNCE, NWJ
from epidemic_simulate_data import solve_sir_sdes


from neural.modules import Mlp
from neural.aggregators import (
    PermutationInvariantImplicitDAD,
    LSTMImplicitDAD,
    ConcatImplicitDAD,
)
from neural.baselines import (
    ConstantBatchBaseline,
    BatchDesignBaseline,
    RandomDesignBaseline,
)
from neural.critics import CriticDotProd, CriticBA

mi_estimator_options = {"NWJ": NWJ, "InfoNCE": InfoNCE}


class SIR_SDE_Simulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, batch_data, device):

        # compute nearest neighbours in time grid
        with torch.no_grad():
            ## Central difference
            nearest = torch.min(
                torch.abs(inputs.reshape(-1, 1) - batch_data["ts"][1:-1]), axis=1
            ).indices
        # extract number of infected from data
        ## Central difference
        y = batch_data["ys"][1:-1][nearest, range(nearest.shape[0])].reshape(-1, 1)

        # y = y.reshape(-1, 1)  # TODO: Make more general
        ctx.save_for_backward(inputs)
        ctx.device = device
        ctx.nearest = nearest

        ## Central difference
        ctx.grads = (batch_data["ys"][2:, :] - batch_data["ys"][:-2, :]) / (
            2 * batch_data["dt"]
        )

        return y

    @staticmethod
    def backward(ctx, grad_output):
        # unpack saved tensors
        inputs = ctx.saved_tensors

        device = ctx.device
        nearest = ctx.nearest
        grads = ctx.grads

        # extract gradients of infected from data
        y_grads = grads[nearest, range(nearest.shape[0])].T
        y_grads = y_grads.reshape(-1, 1)  # TODO: make more general

        # compute the Jacobian
        identity = torch.eye(1, device=device, dtype=torch.float32).reshape(1, 1, 1)
        Jac = torch.mul(identity.repeat(len(y_grads), 1, 1), y_grads[:, None])
        # Jac = y_grads.unsqueeze(-1).unsqueeze(-1)
        # print(Jac.reshape(-1) == y_grads.reshape(-1))
        # print("Jacobian shape:", Jac.shape)
        # compute the Jacobian vector product
        grad_input = Jac.matmul(grad_output[:, :, None]).reshape(-1, 1)
        # print("GRAD INPUT")
        # print(grad_input)
        return grad_input, None, None


class Epidemic(nn.Module):

    """
    Class for the SDE-based SIR model. This version loads in pre-simulated data
    and then access observations corresponding to the emitted design.
    """

    def __init__(
        self,
        design_net,
        T,
        design_transform="iid",
        simdata=None,
        lower_bound=torch.tensor(1e-2),
        upper_bound=torch.tensor(100.0 - 1e-2),
    ):

        super(Epidemic, self).__init__()

        self.p = 2  # dim of latent
        self.design_net = design_net
        self.T = T  # number of experiments
        self.SIMDATA = simdata
        loc = torch.tensor([0.5, 0.1]).log().to(simdata["ys"].device)
        covmat = torch.eye(2).to(simdata["ys"].device) * 0.5 ** 2
        self._prior_on_log_theta = torch.distributions.MultivariateNormal(loc, covmat)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if design_transform == "ts":
            self.transform_designs = self._transform_designs_increasing
        elif design_transform == "iid":
            self.transform_designs = self._transform_designs_independent
        else:
            raise ValueError

    def simulator(self, xi, theta, batch_data):
        # extract data from global dataset
        sim_sir = SIR_SDE_Simulator.apply
        y = sim_sir(xi, batch_data, theta.device)

        return y

    def _get_batch_data(self, indices):
        batch_data = {
            "ys": self.SIMDATA["ys"][:, indices],
            "prior_samples": self.SIMDATA["prior_samples"][indices, :],
            "ts": self.SIMDATA["ts"],
            "dt": self.SIMDATA["dt"],
        }
        return batch_data

    def _transform_designs_increasing(self, xi_untransformed, xi_prev):
        xi_prop = nn.Sigmoid()(xi_untransformed)
        xi = xi_prev + xi_prop * (self.upper_bound - xi_prev)
        return xi

    def _transform_designs_independent(self, xi_untransformed, xi_prev=None):
        xi_prop = nn.Sigmoid()(xi_untransformed)
        xi = self.lower_bound + xi_prop * (self.upper_bound - self.lower_bound)
        return xi

    def _remove_data(self):
        self.SIMDATA = None

    def theta_to_index(self, theta):
        theta_expanded = theta.unsqueeze(1).expand(
            theta.shape[0], self.SIMDATA["prior_samples"].shape[0], theta.shape[1]
        )
        norms = torch.linalg.norm(
            self.SIMDATA["prior_samples"] - theta_expanded, dim=-1
        )
        closest_indices = norms.min(-1).indices
        assert closest_indices.shape[0] == theta.shape[0]
        return closest_indices

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        device = self.SIMDATA["prior_samples"].device
        prior_on_index = dist.Categorical(
            torch.ones(self.SIMDATA["num_samples"], device=device)
        )

        ################################################################################
        # Sample theta
        ################################################################################
        # conditioning should be on the indices:

        indices = pyro.sample("indices", prior_on_index)
        batch_data = self._get_batch_data(indices)

        # helper to 'sample' theta
        def get_theta():
            return batch_data["prior_samples"]

        theta = latent_sample("theta", get_theta)

        y_outcomes = []
        xi_designs = []

        # at t=0 set last design equal to the lower bound
        xi_prev = self.lower_bound

        for t in range(self.T):
            ####################################################################
            # Get a design xi
            ####################################################################
            xi_untransformed = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            # squeeze the first dim (corrresponds to <n>)
            xi = self.transform_designs(
                xi_untransformed=xi_untransformed.squeeze(1), xi_prev=xi_prev,
            )

            ####################################################################
            # Sample y
            ####################################################################
            y = observation_sample(
                f"y{t + 1}", self.simulator, xi=xi, theta=theta, batch_data=batch_data
            )

            ####################################################################
            # Update history
            ####################################################################
            y_outcomes.append(y)
            xi_designs.append(xi_untransformed)  #! pass untransformed

            xi_prev = xi  # set current design as previous for next loop

        del batch_data  # delete manually just in case
        return theta, xi_designs, y_outcomes

    def forward(self, indices):
        """ Run the policy for a given index (corresponding to a latent theta) """
        self.design_net.eval()

        def conditioned_model():
            # indices = self.theta_to_index(theta)
            with pyro.plate_stack("expand_theta_test", [indices.shape[0]]):
                # condition on "theta" (ie the corresponding indices)
                return pyro.condition(self.model, data={"indices": indices})()

        with torch.no_grad():
            theta, designs, observations = conditioned_model()

        return theta, designs, observations

    def eval(self, theta=None, verbose=False):
        """
        Run policy and produce a df with output
        """
        self.design_net.eval()
        # can't do more than one in this form since we (in all likelihood)
        # have one realisation per theta
        n_trace = 1
        if theta is None:
            theta = self._prior_on_log_theta.sample(torch.Size([1])).exp()
            indices = self.theta_to_index(theta)
        else:
            indices = self.theta_to_index(theta)

        output = []
        theta, designs, observations = self.forward(indices)
        for i in range(n_trace):
            run_xis = []
            run_ys = []

            xi_prev = self.lower_bound
            if verbose:
                print("Example run {}".format(i))
                print(f"*True Theta: {theta[i]}*")

            for t in range(self.T):
                xi_untransformed = designs[t][i].detach().cpu()
                xi = self.transform_designs(
                    xi_untransformed=xi_untransformed.squeeze(0), xi_prev=xi_prev,
                )
                xi_prev = xi
                run_xis.append(xi.cpu().reshape(-1))
                y = observations[t][i].detach().cpu().item()
                run_ys.append(y)

                if verbose:
                    print(f"xi{t + 1}: {run_xis[-1][0].data}  y{t + 1}: {y}")

            run_df = pd.DataFrame(torch.stack(run_xis).numpy())
            run_df.columns = [f"xi_{i}" for i in range(xi.shape[0])]
            run_df["observations"] = run_ys
            run_df["order"] = list(range(1, self.T + 1))
            run_df["run_id"] = i + 1
            output.append(run_df)

        return pd.concat(output), theta.cpu().numpy()


def train_model(
    num_steps,
    batch_size,
    num_negative_samples,
    seed,
    lr,
    lr_critic,
    gamma,
    device,
    T,
    hidden_dim,
    encoding_dim,
    critic_arch,
    mi_estimator,
    mlflow_experiment_name,
    design_arch,
    design_transform,
):
    pyro.clear_param_store()

    ### Set up Mlflow logging ### ------------------------------------------------------
    mlflow.set_experiment(mlflow_experiment_name)
    seed = auto_seed(seed)

    #####
    n = 1  # output dim/number of samples per design
    design_dim = (n, 1)  # design is t (time)
    latent_dim = 2
    observation_dim = n

    if lr_critic is None:
        lr_critic = lr
    # Design emitter hidden layer
    des_emitter_HD = encoding_dim // 2

    # History encoder is applied to encoding of both design and critic networks.
    hist_encoder_HD = [8, 64, hidden_dim]

    # These are for critic only:
    latent_encoder_HD = [8, 64, hidden_dim]
    hist_enc_critic_head_HD = encoding_dim // 2

    mlflow.log_param("HD_hist_encoder", str(hist_encoder_HD))
    mlflow.log_param("HD_des_emitter", str(des_emitter_HD))
    mlflow.log_param("HD_latent_encoder", str(latent_encoder_HD))
    mlflow.log_param("HD_hist_enc_critic_head", str(hist_enc_critic_head_HD))

    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("lr_critic", lr_critic)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)

    mlflow.log_param("critic_arch", critic_arch)
    mlflow.log_param("lr_critic", lr_critic)

    mlflow.log_param("design_arch", design_arch)
    mlflow.log_param("mi_estimator", mi_estimator)
    mlflow.log_param("design_transform", design_transform)
    # ----------------------------------------------------------------------------------

    ###################################################################################
    ### Setup design and critic networks
    ###################################################################################
    ### DESIGN NETWORK ###
    history_encoder = Mlp(
        input_dim=[*design_dim, observation_dim],
        hidden_dim=hist_encoder_HD,  # hidden_dim,
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="policy_history_encoder",
    )
    design_emitter = Mlp(
        # iDAD only -> options are sum or cat
        input_dim=encoding_dim * max((T - 1), 1)
        if design_arch == "cat"
        else encoding_dim,
        hidden_dim=des_emitter_HD,
        output_dim=design_dim,
        activation=nn.ReLU(),
        name="policy_design_emitter",
    )
    if design_arch == "sum":
        # iDAD sum aggregator
        design_net = PermutationInvariantImplicitDAD(
            history_encoder,
            design_emitter,
            empty_value=torch.zeros(design_dim, device=device),
        ).to(device)
    elif design_arch == "lstm":
        # iDAD LSTM aggregator
        design_net = LSTMImplicitDAD(
            history_encoder,
            design_emitter,
            empty_value=torch.zeros(design_dim, device=device),
            num_hidden_layers=2,
        ).to(device)
    elif design_arch == "cat":
        # iDAD concat aggregator
        design_net = ConcatImplicitDAD(
            history_encoder,
            design_emitter,
            empty_value=torch.zeros(design_dim, device=device),
            T=T,
        ).to(device)
    elif design_arch == "random":
        # Random baseline
        # no design net, can be independent or TS
        design_net = RandomDesignBaseline(
            design_dim,
            random_designs_dist=torch.distributions.Uniform(
                torch.tensor(-5.0, device=device), torch.tensor(5.0, device=device)
            ),
        ).to(device)
    elif design_arch == "equal_interval":
        # Equal interval baseline
        linspace = torch.linspace(0.01, 0.99, T, dtype=torch.float32)
        mlflow.log_param("init_design", str(list(linspace.numpy())))
        transformed_designs = linspace.to(device).unsqueeze(1)
        const_designs = torch.log(transformed_designs / (1 - transformed_designs))
        design_net = ConstantBatchBaseline(const_designs=const_designs).to(device)
    elif design_arch == "static":
        # Static baseline
        # can be independent or TS
        design_net = BatchDesignBaseline(
            T=T,
            design_dim=design_dim,
            design_init=torch.distributions.Uniform(
                torch.tensor(-5.0, device=device), torch.tensor(5.0, device=device)
            ),
        )
        mlflow.log_param("init_design", "u(-5, 5)")

    ######## CRITIC NETWORK #######
    ## Latent encoder
    critic_latent_encoder = Mlp(
        input_dim=latent_dim,
        hidden_dim=latent_encoder_HD,
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="critic_latent_encoder",
    )
    ## History encoder
    critic_design_obs_encoder = Mlp(
        input_dim=[*design_dim, observation_dim],
        hidden_dim=hist_encoder_HD,
        output_dim=encoding_dim,
        name="critic_design_obs_encoder",
    )
    critic_head = Mlp(
        input_dim=encoding_dim * T if critic_arch == "cat" else encoding_dim,
        hidden_dim=hist_enc_critic_head_HD,
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="critic_head",
    )

    if critic_arch == "cat":
        critic_history_encoder = ConcatImplicitDAD(
            encoder_network=critic_design_obs_encoder,
            emission_network=critic_head,
            empty_value=torch.ones(n, latent_dim, device=device),
            T=T,
        )
    elif critic_arch == "lstm":
        critic_history_encoder = LSTMImplicitDAD(
            encoder_network=critic_design_obs_encoder,
            emission_network=critic_head,
            empty_value=torch.ones(n, latent_dim, device=device),
            num_hidden_layers=2,
        )
    elif critic_arch == "sum":
        critic_history_encoder = PermutationInvariantImplicitDAD(
            encoder_network=critic_design_obs_encoder,
            emission_network=critic_head,
            empty_value=torch.ones(n, latent_dim, device=device),
        )
    else:
        raise ValueError("Invalid critic_arch")

    critic_net = CriticDotProd(
        history_encoder_network=critic_history_encoder,
        latent_encoder_network=critic_latent_encoder,
    ).to(device)

    #######################################################################
    SIMDATA = torch.load("data/sir_sde_data.pt", map_location=device)
    mlflow.log_param("dt", SIMDATA["dt"].cpu().item())
    # will plot evolution of designs of this theta
    test_theta = torch.tensor([[0.60, 0.15]], dtype=torch.float, device=device)

    # print("initilize net")
    # design_net.apply(init_weights)
    epidemic = Epidemic(
        design_net=design_net, T=T, design_transform=design_transform, simdata=SIMDATA,
    )

    def separate_learning_rate(module_name, param_name):
        if module_name == "critic_net":
            return {"lr": lr_critic}
        elif module_name == "design_net":
            return {"lr": lr}
        else:
            raise NotImplementedError()

    optimizer = torch.optim.Adam
    # # Annealed LR. Set factor=1 if no annealing required
    scheduler = pyro.optim.ExponentialLR(
        {"optimizer": optimizer, "optim_args": separate_learning_rate, "gamma": gamma}
    )

    logging_freq = 200
    epoch_size = 1000
    resample_data_epochs = 10
    # we'll re-simulate data every (resample_data_epochs * epoch_size)=10K steps
    mlflow.log_param("resample_data_epochs", resample_data_epochs)

    num_epochs = num_steps // epoch_size
    print("num epochs", num_epochs)

    mi_loss_instance = mi_estimator_options[mi_estimator](
        model=epidemic.model,
        critic=critic_net,
        batch_size=batch_size,
        num_negative_samples=num_negative_samples,
    )

    mlflow.log_param("num_negative_samples", mi_loss_instance.num_negative_samples)
    mlflow.log_param("num_batch_samples", mi_loss_instance.batch_size)

    oed = OED(optim=scheduler, loss=mi_loss_instance)

    outputs_history = []

    num_steps_range = trange(1, num_steps + 1, desc="Loss: 0.000 ")

    epoch_i = 0

    ### Log params:
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    mlflow.log_param("num_params_criticnet", count_parameters(critic_net))
    mlflow.log_param("num_params_designnet", count_parameters(design_net))

    for i in num_steps_range:
        epidemic.train()
        loss = oed.step()
        num_steps_range.set_description("Loss: {:.3f} ".format(loss))

        if (i - 1) % logging_freq == 0:
            loss_eval = oed.evaluate_loss()
            mlflow.log_metric("loss", loss_eval, step=i)
            df, latents = epidemic.eval(theta=test_theta, verbose=False)
            df["step"] = i
            outputs_history.append(df)

        if i % epoch_size == 0:
            epoch_i += 1
            scheduler.step()  # loss_eval
            # each 10th epoch resample
            if epoch_i % resample_data_epochs == 0 and epoch_i < num_epochs:
                print("resampling SIMDATA")
                epidemic._remove_data()
                del SIMDATA
                SIMDATA = solve_sir_sdes(num_samples=120000, device=device, grid=10000)
                SIMDATA = {
                    key: (
                        value.to(device) if isinstance(value, torch.Tensor) else value
                    )
                    for key, value in SIMDATA.items()
                }
                epidemic.SIMDATA = SIMDATA

    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")

    pd.concat(outputs_history).reset_index().to_csv(f"mlflow_outputs/designs_hist.csv")
    mlflow.log_artifact(f"mlflow_outputs/designs_hist.csv", artifact_path="designs")

    epidemic.eval(theta=test_theta, verbose=True)
    epidemic._remove_data()
    # store params, metrics and artifacts to mlflow ------------------------------------
    print("Storing model to MlFlow... ", end="")
    mlflow.pytorch.log_model(epidemic.cpu(), "model")
    print("Storing critic network to MlFlow... ", end="")
    mlflow.pytorch.log_model(critic_net.cpu(), "critic")

    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts"
    print(f"Model and critic sotred in {model_loc}. Done.")
    mlflow.log_param("status", "complete")

    return epidemic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iDAD: SDE-Based SIR Model")
    parser.add_argument("--num-steps", default=100000, type=int)
    parser.add_argument("--num-batch-samples", default=512, type=int)
    parser.add_argument("--num-negative-samples", default=511, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--lr-critic", default=None, type=float)
    parser.add_argument("--gamma", default=0.96, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num-experiments", default=5, type=int)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--encoding-dim", default=32, type=int)
    parser.add_argument("--mi-estimator", default="InfoNCE", type=str)
    parser.add_argument(
        "--design-transform", default="ts", type=str, choices=["ts", "iid"]
    )
    # cat, lstm (suitable for ts) or sum (suitable for iid)
    parser.add_argument(
        "--critic-arch", default="lstm", type=str, choices=["cat", "sum", "lstm"]
    )
    # iDAD: <sum> or <lstm>
    # Baselines: choice between  <static>, <equal_interval> and <random>
    parser.add_argument(
        "--design-arch",
        default="lstm",
        type=str,
        choices=["sum", "lstm", "static", "equal_interval", "random"],
    )

    parser.add_argument("--mlflow-experiment-name", default="epidemic", type=str)
    args = parser.parse_args()

    train_model(
        num_steps=args.num_steps,
        batch_size=args.num_batch_samples,
        num_negative_samples=args.num_negative_samples,
        seed=args.seed,
        lr=args.lr,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        device=args.device,
        T=args.num_experiments,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        critic_arch=args.critic_arch,
        mi_estimator=args.mi_estimator,
        mlflow_experiment_name=args.mlflow_experiment_name,
        design_arch=args.design_arch,
        design_transform=args.design_transform,
    )
