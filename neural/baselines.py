import torch
from torch import nn
from neural.modules import LazyDelta


########## Baselines #############
# this covers MINEBED and SG-BOED (ACE estimator with prior as proposal)
class DesignBaseline(nn.Module):
    def __init__(self, design_dim):
        super().__init__()
        self.register_buffer("prototype", torch.zeros(design_dim))

    def lazy(self, *design_obs_pairs):
        def delayed_function():
            return self.forward(*design_obs_pairs)

        lazy_delta = LazyDelta(
            delayed_function, self.prototype, event_dim=self.prototype.dim()
        )
        return lazy_delta


class BatchDesignBaseline(DesignBaseline):
    """
    Batch design baseline: learns T constants.

    - If trained with InfoNCE bound, this is the SG-BOED static baseline.
    - If trained with the NWJ bound, this is the MINEBED static baselines.
    """

    def __init__(
        self,
        T,
        design_dim,
        output_activation=nn.Identity(),
        design_init=torch.distributions.Normal(0, 0.5),
    ):
        super().__init__(design_dim)
        self.designs = nn.ParameterList(
            [
                nn.Parameter(design_init.sample(torch.zeros(design_dim).shape))
                for i in range(T)
            ]
        )
        self.output_activation = output_activation

    def forward(self, *design_obs_pairs):
        j = len(design_obs_pairs)
        return self.output_activation(self.designs[j])


class ConstantBatchBaseline(DesignBaseline):
    def __init__(self, design_dim, const_designs_list):
        super().__init__(design_dim)
        self.designs = const_designs_list

    def forward(self, *design_obs_pairs):
        j = len(design_obs_pairs)
        return self.designs[j]


class RandomDesignBaseline(DesignBaseline):
    def __init__(self, design_dim, random_designs_dist=None):
        super().__init__(design_dim)
        if random_designs_dist is None:
            random_designs_dist = torch.distributions.Normal(
                torch.zeros(design_dim), torch.ones(design_dim)
            )
        self.random_designs_dist = random_designs_dist

    def forward(self, *design_obs_pairs):
        return self.random_designs_dist.sample()
