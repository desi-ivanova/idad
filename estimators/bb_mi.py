import math

import torch

import pyro
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import warn_if_nan
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand


class BlackBoxMutualInformation(object):
    def __init__(
        self, model, critic, batch_size, num_negative_samples, data_source=None
    ):
        self.model = model
        self.critic = critic
        self._batch_size_arg = batch_size
        self._num_negative_samples_arg = num_negative_samples
        self.data_source = data_source

        if critic.critic_type == "separable":
            # sample at least batch_size = num_negative_samples + 1,
            # so we get num_negative_samples off-diagonal terms.
            # this is done mainly for eval purposes where we want few samples in batch
            self.batch_size = max(num_negative_samples + 1, self._batch_size_arg)
            # max out num_negative_samples
            self.num_negative_samples = self.batch_size - 1
            self.get_scores = self._get_scores_separable_critic
        elif critic.critic_type == "joint":
            # num_negative_samples should be less than batch_size-1 maybe (?)
            self.num_negative_samples = min(num_negative_samples, batch_size - 1)
            self.batch_size = batch_size
            self.get_scores = self._get_scores_joint_critic
        else:
            raise ValueError("Invalid critic type.")

    def _vectorized(self, fn, *shape, name="vectorization_plate"):
        """
        Wraps a callable inside an outermost :class:`~pyro.plate` to parallelize
        MI computation over `num_particles`, and to broadcast batch shapes of
        sample site functions in accordance with the `~pyro.plate` contexts
        within which they are embedded.
        :param fn: arbitrary callable containing Pyro primitives.
        :return: wrapped callable.
        """

        def wrapped_fn(*args, **kwargs):
            with pyro.plate_stack(name, shape):
                return fn(*args, **kwargs)

        return wrapped_fn

    def get_primary_rollout(self, args, kwargs, graph_type="flat", detach=False):
        """
        sample data: batch_size number of examples -> return trace
        """
        if self.data_source is None:
            model_v = self._vectorized(
                self.model, self.batch_size, name="outer_vectorization"
            )
        else:
            data = next(self.data_source)
            model = pyro.condition(
                self._vectorized(model, self.batch_size, name="outer_vectorization"),
                data=data,
            )

        trace = poutine.trace(model_v, graph_type=graph_type).get_trace(*args, **kwargs)
        if detach:
            # what does the detach do?
            trace.detach_()
        trace = prune_subsample_sites(trace)
        return trace

    def _get_data(self, args, kwargs, graph_type="flat", detach=False):
        # esample a trace and xtract the relevant data from it
        trace = self.get_primary_rollout(args, kwargs, graph_type, detach)
        designs = [
            node["value"]
            for node in trace.nodes.values()
            if node.get("subtype") == "design_sample"
        ]
        observations = [
            node["value"]
            for node in trace.nodes.values()
            if node.get("subtype") == "observation_sample"
        ]
        latents = [
            node["value"]
            for node in trace.nodes.values()
            if node.get("subtype") == "latent_sample"
        ]
        latents = torch.cat(latents, axis=-1)
        return (latents, *zip(designs, observations))

    def _get_scores_separable_critic(
        self, args, kwargs, graph_type="flat", detach=False
    ):
        data = self._get_data(args, kwargs, graph_type, detach)
        # Critics return two matrcies: <joint_scores> and <product_scores>
        # For separable critics: Both of these have shape [batch_sahpe, batch_shape]
        ## joint_scores: only diag is non-zero
        ## product_scores: diag is all zeros
        # Rows are batch examples -> logsumexp-ing should be along dim=1
        return self.critic(*data)

    def _get_scores_joint_critic(self, args, kwargs, graph_type="flat", detach=False):

        latents, *history = self._get_data(args, kwargs, graph_type, detach)
        # generate negative examples by shuffling sampled latents:
        latents_shuffle = [
            latents[torch.randperm(self.batch_size)]
            for _ in range(self.num_negative_samples)
        ]
        # want columns to be latents and rows be batches, so need to cat on dim 1 (!)
        # so that it is consistent with sep critic and can logsumexp along dim=1
        latents_combined = torch.stack([latents] + latents_shuffle, dim=1)
        # Critics return two matrcies: <joint_scores> and <product_scores>
        # For joint critics: both have shape [batch_sahpe, 1 + num_negative_samples]
        ## joint_scores: only first columns is non-zero (positive examples)
        ## product_scores: first column is all zeros
        # Rows are batch examples -> logsumexp-ing should be along dim=1
        return self.critic(latents_combined, *history)


class InfoNCE(BlackBoxMutualInformation):
    def __init__(
        self, model, critic, batch_size, num_negative_samples, data_source=None
    ):
        super().__init__(
            model=model,
            critic=critic,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
            data_source=data_source,
        )

    def differentiable_loss(self, *args, **kwargs):
        pyro.module("critic_net", self.critic)  #!!

        joint_scores, product_scores = self.get_scores(args, kwargs)
        # if critic is separable: joint_scores = matrix with all 0s except the diag.
        # if critic is joint: joint_scores = matrix with all 0s except the first col,
        # so summing and dividing by batch_size will work in both cases.
        ### DON'T do .mean(): we have 0s for the negative examples ###
        joint_term = joint_scores[: self._batch_size_arg].sum() / self._batch_size_arg

        # if critic is separable: product_scores = matrix with all 0s except the diag.
        # if critic is joint: product_scores = matrix with all 0s except the first col,
        # so summing the joint and product matrices will work in both cases. Here .mean
        # is across batch, so we still need to substract log(num_negative + 1),
        # which is done in .loss() method
        product_term = (
            (joint_scores + product_scores)[: self._batch_size_arg]
            .logsumexp(dim=1)
            .mean()
        )
        loss = product_term - joint_term

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using InfoNCE bound == -EIG
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant - math.log(self.num_negative_samples + 1)


class NWJ(BlackBoxMutualInformation):
    def __init__(
        self, model, critic, batch_size, num_negative_samples, data_source=None
    ):
        super().__init__(
            model=model,
            critic=critic,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
            data_source=data_source,
        )

    def differentiable_loss(self, *args, **kwargs):
        pyro.module("critic_net", self.critic)  # !!

        joint_scores, product_scores = self.get_scores(args, kwargs)
        ### DON'T do .mean(): we have 0s for the negative examples ###
        joint_term = joint_scores.sum() / self.batch_size
        # Careful with how many elements we have and remove the effect of the 0s:
        product_term = (
            (
                # we have 0-s at the positive examples (of which we have batch_size)
                # so at the end we have batch_size*exp(0) extra which we substract here
                product_scores.exp().sum()
                - self.batch_size
            )
            # divide by e
            * math.exp(-1)
            # average -> there are a total of batch_size * num_negative_samples entries:
            / (self.batch_size * self.num_negative_samples)
        )
        loss = product_term - joint_term

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using NWJ bound  == -EIG proxy
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant


class BarberAgakov(BlackBoxMutualInformation):
    def __init__(self, model, critic, batch_size, prior_entropy, **kwargs):
        super().__init__(
            model=model, critic=critic, batch_size=batch_size, num_negative_samples=0
        )
        self.prior_entropy = prior_entropy

    def differentiable_loss(self, *args, **kwargs):
        pyro.module("critic_net", self.critic)  # !!
        latents, *history = self._get_data(args, kwargs)

        log_probs_q = self.critic(latents, *history)

        loss = -log_probs_q.mean()
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using the BA lower bound == -EIG
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant - self.prior_entropy
