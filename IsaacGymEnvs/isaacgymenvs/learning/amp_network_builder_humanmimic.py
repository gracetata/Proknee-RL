# Copyright (c) 2018-2023, NVIDIA Corporation
# Parallel AMP builder: shared trunk + imitation logit + style logit (HumanMimic-style blending).

from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn

DISC_LOGIT_INIT_SCALE = 1.0


class AMPBuilderHumanMimic(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if not self.space_config["learn_sigma"]:
                    actions_num = kwargs.get("actions_num")
                    sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
                    self.sigma = nn.Parameter(
                        torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False
                    )
                    sigma_init(self.sigma)

            amp_input_shape = kwargs.get("amp_input_shape")
            self._build_disc(amp_input_shape)

        def load(self, params):
            super().load(params)

            self._disc_units = params["disc"]["units"]
            self._disc_activation = params["disc"]["activation"]
            self._disc_initializer = params["disc"]["initializer"]

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            return self._disc_logits(disc_mlp_out)

        def eval_style_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            return self._style_logits(disc_mlp_out)

        def get_disc_logit_weights(self):
            return torch.cat([torch.flatten(self._disc_logits.weight), torch.flatten(self._style_logits.weight)])

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))
            weights.append(torch.flatten(self._disc_logits.weight))
            weights.append(torch.flatten(self._style_logits.weight))
            return weights

        def _build_disc(self, input_shape):
            mlp_args = {
                "input_size": input_shape[0],
                "units": self._disc_units,
                "activation": self._disc_activation,
                "dense_func": torch.nn.Linear,
            }
            self._disc_mlp = self._build_mlp(**mlp_args)

            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)
            self._style_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            for head in (self._disc_logits, self._style_logits):
                torch.nn.init.uniform_(head.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
                torch.nn.init.zeros_(head.bias)

    def build(self, name, **kwargs):
        net = AMPBuilderHumanMimic.Network(self.params, **kwargs)
        return net
