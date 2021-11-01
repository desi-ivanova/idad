import torch
from torch import nn

from neural.modules import LazyDelta


class ImplicitDeepAdaptiveDesign(nn.Module):
    def __init__(self, encoder_network, emission_network, empty_value):
        super().__init__()
        self.encoder = encoder_network
        ## [!] store encoding dim
        self.encoding_dim = encoder_network.output_dim
        self.emitter = (
            emission_network if emission_network is not None else nn.Identity()
        )
        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))
        # self.empty_value = empty_value

    def lazy(self, *design_obs_pairs):
        def delayed_function():
            return self.forward(*design_obs_pairs)

        lazy_delta = LazyDelta(
            delayed_function, self.prototype, event_dim=self.prototype.dim()
        )
        return lazy_delta

    def forward(self,):
        raise NotImplementedError()


class PermutationInvariantImplicitDAD(ImplicitDeepAdaptiveDesign):
    def __init__(
        self, encoder_network, emission_network, empty_value, self_attention_layer=None
    ):
        super().__init__(
            encoder_network=encoder_network,
            emission_network=emission_network,
            empty_value=empty_value,
        )
        self.selfattention_layer = (
            self_attention_layer if self_attention_layer is not None else nn.Identity()
        )

    def sum_history_encodings(self, *design_obs_pairs):
        # encode available design-obs pairs, h_t, and stack the representations
        # dimension is: [batch_size, t, encoding_dim]
        stacked = torch.stack(
            [
                self.encoder(design, obs)
                for idx, (design, obs) in enumerate(design_obs_pairs)
            ],
            dim=1,
        )
        # apply attention (or identity if attention=None)
        stacked = self.selfattention_layer(stacked)
        # sum-pool the resulting encodings across t (dim=1)
        # dimension is: [batch_size, encoding_dim]
        sum_encoding = stacked.sum(dim=1)
        return sum_encoding

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            # For efficiency: learn the first design separately, i.e. do not pass a
            # vector (e.g. of 0s) through the emitter network.
            # !This doesn't affect critic net, since len(design_obs_pairs) is never 0.
            output = self.empty_value
            #### To pass a zero vector though the emitter, use this: ###
            # zero_vec = self.empty_value.new_zeros(self.encoding_dim)
            # output = self.emitter(zero_vec)

        else:
            sum_encoding = self.sum_history_encodings(*design_obs_pairs)
            output = self.emitter(sum_encoding)
        return output


class LSTMImplicitDAD(ImplicitDeepAdaptiveDesign):
    def __init__(
        self, encoder_network, emission_network, empty_value, num_hidden_layers=2
    ):
        super().__init__(encoder_network, emission_network, empty_value)
        self.lstm_net = nn.LSTM(
            self.encoding_dim, self.encoding_dim, num_hidden_layers, batch_first=True
        )

    def lstm_history_encodings(self, *design_obs_pairs):
        # Input to LSTM should be [batch, seq, feature]
        if len(design_obs_pairs) == 0:
            # pass zeros to the LSTM if no history is available yet
            stacked = self.empty_value.new_zeros(1, 1, self.encoding_dim)
        else:
            # encode available design-obs pairs, h_t, and stack the representations
            # dimension is: [batch_size, t, encoding_dim]
            stacked = torch.stack(
                [
                    self.encoder(design, obs, t=[idx + 1])
                    for idx, (design, obs) in enumerate(design_obs_pairs)
                ],
                dim=1,
            )
        # keep the last state
        _, (h_n, c_n) = self.lstm_net(stacked)
        # return the hidden state from the last layer
        # dimension [batch_size, encoding_dim]
        return h_n[-1]

    def forward(self, *design_obs_pairs):
        lstm_encoding = self.lstm_history_encodings(*design_obs_pairs)
        return self.emitter(lstm_encoding)


class ConcatImplicitDAD(ImplicitDeepAdaptiveDesign):
    def __init__(self, encoder_network, emission_network, T, empty_value=None):
        super().__init__(encoder_network, emission_network, empty_value)

        def _nopad(x):
            return x

        def _pad(x):
            # padding = (0, self._target_dim - x.shape[-1]) means no padding on the
            # left, target minus what number of elements already have on the right
            pad = torch.nn.ConstantPad1d(
                padding=(0, self._target_dim - x.shape[-1]), value=0.0
            )
            return pad(x)

        self.T = T
        self._target_dim = self.encoding_dim * max(T - 1, 1)
        self.apply_padding = _nopad

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            # if true -> this is a design network -> pad
            stacked = self.empty_value.new_zeros(self._encoding_dim)
            self.apply_padding = _pad
        else:
            cat_input = torch.cat(
                [
                    self.encoder(design, obs)
                    for idx, (design, obs) in enumerate(design_obs_pairs)
                ],
                dim=-1,
            )
        return self.emitter(self.apply_padding(cat_input))
