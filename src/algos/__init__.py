# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship.
#
# isir-internship is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Package description"""

__version__ = "0.1.0"
__author__ = "Paul Chambaz"


# from .afu3 import AFU
from .afu import AFU
from .afup import AFUP
from .afutqc2 import AFUTQC
from .mlp import MLP
from .msac import MSAC
from .ndtop import NDTOP
from .rl_algo import RLAlgo
from .sac import SAC
from .simple_right import SimpleRight
from .top import TOP
from .tqc import TQC

__all__ = [
    "AFU",
    "AFU1A",
    "AFUP",
    "AFUTQC",
    "MLP",
    "MSAC",
    "SAC",
    "TOP",
    "TQC",
    "Algo",
    "RLAlgo",
    "SimpleRight",
]
