# Copyright (c) 2018-2023, NVIDIA Corporation
# ModelAMPContinuous with dual AMP heads (imitation + style).

import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class ModelAMPContinuousHumanMimic(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        net = self.network_builder.build("amp_humanmimic", **config)
        for name, _ in net.named_parameters():
            print(name)

        obs_shape = config["input_shape"]
        normalize_value = config.get("normalize_value", False)
        normalize_input = config.get("normalize_input", False)
        value_size = config.get("value_size", 1)

        return self.Network(
            net,
            obs_shape=obs_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            value_size=value_size,
        )

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            result = super().forward(input_dict)

            if is_train:
                amp_obs = input_dict["amp_obs"]
                result["disc_agent_logit"] = self.a2c_network.eval_disc(amp_obs)
                result["style_agent_logit"] = self.a2c_network.eval_style_disc(amp_obs)

                amp_obs_replay = input_dict["amp_obs_replay"]
                result["disc_agent_replay_logit"] = self.a2c_network.eval_disc(amp_obs_replay)
                result["style_agent_replay_logit"] = self.a2c_network.eval_style_disc(amp_obs_replay)

                amp_demo_obs = input_dict["amp_obs_demo"]
                result["disc_demo_logit"] = self.a2c_network.eval_disc(amp_demo_obs)
                result["style_demo_logit"] = self.a2c_network.eval_style_disc(amp_demo_obs)

            return result
