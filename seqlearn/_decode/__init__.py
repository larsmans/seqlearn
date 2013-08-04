# Copyright Lars Buitinck 2013.

"""Decoding (inference) algorithms."""

import numpy as np

from .bestfirst import bestfirst
from .viterbi import viterbi

DECODERS = {"bestfirst": bestfirst,
            "viterbi": viterbi}
