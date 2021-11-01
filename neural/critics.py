from collections import OrderedDict

import torch
from torch import nn


## MI critics
class CriticDotProd(nn.Module):
    """
    Separable critic

    returns:
    scores_joint: tensor of shape [batch_size, batch_size] where only non-zero terms are on the diagonal
    scores_prod: tensor of shape [batch_size, batch_size] where the diagonal terms are all zeros
    """

    def __init__(
        self, history_encoder_network, latent_encoder_network,
    ):
        super().__init__()
        self.critic_type = "separable"
        self.history_encoder_network = history_encoder_network
        self.latent_encoder_network = latent_encoder_network

    def forward(self, latent, *design_obs_pairs):
        history_encoding = self.history_encoder_network(*design_obs_pairs)
        latent_encoding = self.latent_encoder_network(latent)

        pos_mask = torch.eye(history_encoding.shape[0], device=history_encoding.device)
        neg_mask = 1.0 - pos_mask

        # we get (N^2 - batch_size) terms for "free" by reusing sampled data
        score_matrix = torch.matmul(history_encoding, latent_encoding.T)
        scores_joint = score_matrix * pos_mask
        scores_prod = score_matrix * neg_mask
        return scores_joint, scores_prod


class CriticJointNetwork(nn.Module):
    """Joint critic
    fc_layers : nn.Sequential instance, should return output of size 1
        if not specified, default is to do linear -> Relu -> output
    returns:
    scores_joint: tensor of shape [batch_size, 1 + num_negative_samples]
        The first column contains the positive examples scores; the rest are 0.
    scores_prod: tensor of shape [batch_size, 1 + num_negative_samples];
        The first column is 0s; the rest contain the negative examples scores.
    """

    def __init__(
        self, history_encoder_network, latent_encoder_network, head_layer=None
    ):
        super().__init__()
        self.critic_type = "joint"
        self.history_encoder_network = history_encoder_network
        self.latent_encoder_network = latent_encoder_network

        if head_layer is not None:
            self.head_layer = head_layer
        else:
            ## [!] relying on encoder netowkrs having .encoding_dim attributes ##
            input_dim = (
                latent_encoder_network.encoding_dim
                + history_encoder_network.encoding_dim
            )

            self.head_layer = nn.Sequential(
                OrderedDict(
                    [
                        ("critic_l1", nn.Linear(input_dim, 512)),
                        ("critic_relu1", nn.ReLU()),
                        # ("critic_l2", nn.Linear(2 * input_dim, input_dim)),
                        # ("critic_relu2", nn.ReLU()),
                        ("critic_output", nn.Linear(512, 1)),
                    ]
                )
            )

    def forward(self, latent, *design_obs_pairs):
        # Latents is a tensor of dim [batch_samples, negativesamples + 1, encodning dim]
        latent_encoding = self.latent_encoder_network(latent)
        history_encoding = self.history_encoder_network(*design_obs_pairs)
        # expand the middle dimension (i.e. negative samples)
        history_encoding = history_encoding.unsqueeze(1).expand(latent_encoding.shape)

        inputs = torch.cat([history_encoding, latent_encoding], axis=-1)
        # remove last dim (output (ie score_matrix last dim) is of size 1):
        score_matrix = self.head_layer(inputs).squeeze(-1)

        pos_mask = score_matrix.new_zeros(score_matrix.shape)
        pos_mask[:, 0] = 1.0  # this is the unshuffled latent
        neg_mask = 1.0 - pos_mask

        scores_joint = score_matrix * pos_mask
        scores_prod = score_matrix * neg_mask

        return scores_joint, scores_prod


class CriticBA(nn.Module):
    """Barber Agakov variational critic
    fc_layers : nn.Sequential instance, should return output of size 1
        if not specified, default is to do linear -> Relu -> output
    returns:
    scores_joint: tensor of shape [batch_size, 1 + num_negative_samples]
        The first column contains the positive examples scores; the rest are 0.
    scores_prod: tensor of shape [batch_size, 1 + num_negative_samples];
        The first column is 0s; the rest contain the negative examples scores.
    """

    def __init__(
        self,
        latent_dim,
        history_encoder_network,
        # latent_encoder_network,
        head_layer_mean=None,
        head_layer_sd=None,
    ):
        super().__init__()
        self.critic_type = "joint"
        self.history_encoder_network = history_encoder_network
        # self.latent_encoder_network = latent_encoder_network

        ## [!] relying on encoder networkrs having .encoding_dim attributes ##
        input_dim = history_encoder_network.encoding_dim
        ## [!] relying on latent encoder networkr having .input_dim_flat attribute ##
        # this is the dimension of the latent
        # this is to set the output dimension equal to the dim of the latent.
        def _reshape_input(x):
            return x.flatten(-2)

        def _id(x):
            return x

        if isinstance(latent_dim, int):
            latent_dim_flat = latent_dim
            self._prepare_input = _id
        else:
            latent_dim_flat = latent_dim[0] * latent_dim[1]
            self._prepare_input = _reshape_input

        if head_layer_mean is not None:
            self.head_layer_mean = head_layer_mean
        else:
            self.head_layer_mean = nn.Sequential(
                OrderedDict(
                    [
                        ("critic_ba_l1_mean", nn.Linear(input_dim, 512)),
                        ("critic_ba_relu1_mean", nn.ReLU()),
                        ("critic_ba_output_mean", nn.Linear(512, latent_dim_flat)),
                    ]
                )
            )
        if head_layer_sd is not None:
            self.head_layer_sd = head_layer_sd
        else:
            self.head_layer_sd = nn.Sequential(
                OrderedDict(
                    [
                        ("critic_ba_l1_sd", nn.Linear(input_dim, 512)),
                        ("critic_ba_relu1_sd", nn.ReLU()),
                        ("critic_ba_output_sd", nn.Linear(512, latent_dim_flat)),
                        ("critic_ba_softplus", nn.Softplus()),
                    ]
                )
            )

    def get_variational_params(self, *design_obs_pairs):
        history_encoding = self.history_encoder_network(*design_obs_pairs)
        mean = self.head_layer_mean(history_encoding)
        sd = 1e-5 + self.head_layer_sd(history_encoding)
        return mean, sd

    def forward(self, latent, *design_obs_pairs):
        latent_flat = self._prepare_input(latent)
        mean, sd = self.get_variational_params(*design_obs_pairs)
        log_probs_q = (
            torch.distributions.Normal(loc=mean, scale=sd)
            .log_prob(latent_flat)
            .sum(axis=-1)
        )

        return log_probs_q
