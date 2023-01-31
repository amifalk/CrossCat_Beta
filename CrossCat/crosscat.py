import pandas as pd
import numpy as np

from state import State

import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy
from fastprogress.fastprogress import progress_bar


class CrossCat:
    def __init__(self, data, dists=None, alpha=None, views=None, cats=None):
        self.data = data

        if dists is None:
            dists = self.infer_dists()

        self.curr = State(data.view(), dists, alpha, views, cats)
        self.states = []
        self.liks = []

        # posterior estimates (can lazily construct these)

    def infer_dists(self):
        """guess the distributions that each column belongs to"""
        dists = []

        for col in self.data.T:
            if np.all((col == 1) | (col == 0)):
                dists.append("bern")
            else:
                dists.append("normal")

        return dists

    def run(self, sweeps=10, sample_every=1):
        progress = progress_bar(range(sweeps))

        for sweep in progress:
            if sweep % sample_every == 0:
                sample = deepcopy(self.curr)

                self.states.append(sample)
                self.liks.append(sample.log_lik())

            self.curr.transition()

    def best_fit(self):
        # TODO: faster if I insert this in the run code, but this is easier for now
        max_lik = max(self.liks)
        index = self.liks.index(max_lik)

        return self.states[index]

    def plot_liks(self):
        sns.lineplot(self.liks)
        plt.title(f"Model Log Likelihood/Time")
        plt.xlabel("Iterations")
        plt.ylabel("Log Likelihood")
        plt.show()

    def save(path):
        # TODO
        pass

    def load(path):
        # TODO
        pass

    # accept pandas dataframes ... convert
