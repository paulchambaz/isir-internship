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

from .afu import AFU
from .afup import AFUP
from .rl_algo import RLAlgo
from .sac import SAC
from .simple_right import SimpleRight
from .tqc import TQC

__all__ = ["AFU", "AFUP", "SAC", "TQC", "Algo", "RLAlgo", "SimpleRight"]
