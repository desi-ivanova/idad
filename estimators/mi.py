import math

import torch

import pyro
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import warn_if_nan
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand


class MutualInformation(object):
    def __init__(self, model, batch_size, data_source=None):
        self.model = model
        self.batch_size = batch_size
        self.data_source = data_source

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
        if self.data_source is None:
            model_v = self._vectorized(
                self.model, self.batch_size, name="outer_vectorization"
            )

        else:
            data = next(self.data_source)
            model_v = pyro.condition(
                self._vectorized(
                    self.model, self.batch_size, name="outer_vectorization"
                ),
                data=data,
            )
        trace = poutine.trace(model_v, graph_type=graph_type).get_trace(*args, **kwargs)
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()
        return trace


class PriorContrastiveEstimation(MutualInformation):
    def __init__(
        self, model, batch_size=100, num_negative_samples=10, data_source=None
    ):
        super().__init__(model=model, batch_size=batch_size, data_source=data_source)
        self.num_negative_samples = num_negative_samples

    def compute_observation_log_prob(self, trace):
        """
        Computes the log probability of observations given latent variables and designs.
        :param trace: a Pyro trace object
        :return: the log prob tensor
        """
        return sum(
            node["log_prob"]
            for node in trace.nodes.values()
            if node.get("subtype") == "observation_sample"
        )

    def get_contrastive_rollout(
        self,
        trace,
        args,
        kwargs,
        existing_vectorization,
        graph_type="flat",
        detach=False,
    ):
        sampled_observation_values = {
            name: lexpand(node["value"], self.num_negative_samples)
            for name, node in trace.nodes.items()
            if node.get("subtype") in ["observation_sample", "design_sample"]
        }
        conditional_model = self._vectorized(
            pyro.condition(self.model, data=sampled_observation_values),
            self.num_negative_samples,
            *existing_vectorization,
            name="inner_vectorization",
        )
        trace = poutine.trace(conditional_model, graph_type=graph_type).get_trace(
            *args, **kwargs
        )
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()

        return trace

    def differentiable_loss(self, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, args, kwargs, [self.batch_size]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)

        obs_log_prob_combined = torch.cat(
            [lexpand(obs_log_prob_primary, 1), obs_log_prob_contrastive]
        ).logsumexp(0)

        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using prior contrastive estimation; == -EIG proxy
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant - math.log(self.num_negative_samples + 1)


class NestedMonteCarloEstimation(PriorContrastiveEstimation):
    def differentiable_loss(self, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, args, kwargs, [self.batch_size]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)
        obs_log_prob_combined = obs_log_prob_contrastive.logsumexp(0)
        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using prior contrastive estimation
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant - math.log(self.num_negative_samples)


class PriorContrastiveEstimationIntegrateNonTarget(PriorContrastiveEstimation):
    def __init__(
        self,
        model,
        batch_size=1,
        num_negative_samples=int(5e5),
        num_integration_samples=512,
        data_source=None,
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
            data_source=data_source,
        )
        self.num_integration_samples = num_integration_samples

    def get_inner_intergal_rollout(
        self,
        trace,
        args,
        kwargs,
        existing_vectorization,
        graph_type="flat",
        detach=False,
    ):
        # condition on data and latents
        # resample those that weren't labeles as latent_sample
        # (e.g. beta, log_k in the mixed effects model)
        sampled_observation_and_latent_values = {
            name: lexpand(node["value"], self.num_integration_samples)
            for name, node in trace.nodes.items()
            if node.get("subtype")
            in ["observation_sample", "design_sample", "latent_sample"]
        }

        conditional_model = self._vectorized(
            pyro.condition(self.model, data=sampled_observation_and_latent_values),
            self.num_integration_samples,
            *existing_vectorization,
            name="inner_integration_vectorization",
        )
        trace = poutine.trace(conditional_model, graph_type=graph_type).get_trace(
            *args, **kwargs
        )
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()

        return trace

    def differentiable_loss(self, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(args, kwargs)
        primary_trace_integrated = self.get_inner_intergal_rollout(
            primary_trace, args, kwargs, [self.batch_size]
        )
        # average across <integration_samples> dimension
        obs_log_prob_primary = self.compute_observation_log_prob(
            primary_trace_integrated
        ).logsumexp(0) - math.log(self.num_integration_samples)

        # denominator gets integrated across everything - no need to resample the
        # non-targets - just sample everything and average [???]
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace,
            args,
            kwargs,
            # [self.num_integration_samples, self.batch_size],
            [self.batch_size],
        )
        # denominator gets integrated across everything - no need to resample the
        # non-targets - just sample everything and average [???]
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)

        obs_log_prob_combined = torch.cat(
            [lexpand(obs_log_prob_primary, 1), obs_log_prob_contrastive]
        ).logsumexp(0)
        ## Upper bound
        # obs_log_prob_combined = obs_log_prob_contrastive.logsumexp(0)
        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss


class NestedMonteCarloEstimationIntegrateNonTarget(
    PriorContrastiveEstimationIntegrateNonTarget
):
    def differentiable_loss(self, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(args, kwargs)
        primary_trace_integrated = self.get_inner_intergal_rollout(
            primary_trace, args, kwargs, [self.batch_size]
        )
        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        # average across <integration_samples> dimension
        obs_log_prob_primary_integrated = self.compute_observation_log_prob(
            primary_trace_integrated
        )
        primary_obs_log_prob_combined = torch.cat(
            [lexpand(obs_log_prob_primary, 1), obs_log_prob_primary_integrated]
        ).logsumexp(0) - math.log(self.num_integration_samples + 1)

        # denominator gets integrated across everything - no need to resample the
        # non-targets - just sample everything and average
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace,
            args,
            kwargs,
            # [self.num_integration_samples, self.batch_size],
            [self.batch_size],
        )
        # denominator gets integrated across everything - no need to resample the
        # non-targets - just sample everything and average
        obs_log_prob_contrastive = self.compute_observation_log_prob(
            contrastive_trace
        ).logsumexp(0)

        loss = (obs_log_prob_contrastive - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using prior contrastive estimation
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant - math.log(self.num_negative_samples)
