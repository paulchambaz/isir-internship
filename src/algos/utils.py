# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import torch
from torch import nn


def soft_update_target(
    target_net: nn.Module, source_net: nn.Module, tau: float
) -> None:
    """
    Performs soft update on a target network from a source network.

    Args:
        target_net: Target network to update
        source_net: Source network to copy from
        tau: Soft update coefficient (0 < tau <= 1)
    """
    with torch.no_grad():
        for target_param, source_param in zip(
            target_net.parameters(), source_net.parameters(), strict=True
        ):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )
