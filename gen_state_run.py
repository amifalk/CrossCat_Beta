import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
import seaborn as sns

from state import State

# Fig 4, example (a) from crosscat 2011
data = np.array(
    [
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0],
    ]
)

n_obs, n_dims = data.shape

my_state = State(data, alpha=default_rng().gamma(1, 1))
my_state.transition(20)


