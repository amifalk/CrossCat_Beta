import numpy as np

from crp import CRP
import utils


class View(CRP):
    def __init__(self, alpha, dims):
        super().__init__(alpha)
        self.dims = dims  # set of dimensions

    def transition_clusters(self):
        for row in range(self.N):  # TODO does this have to be random?
            cluster_i = self.cluster_of(row)

            # calc priors:
            priors = self.calc_transition_log_probs(cluster_i)

            # calc likes:
            log_liks = np.zeros(len(self.elements) + 1)

            for dim in self.dims:
                log_liks += dim.calc_post_bern_row(row, self.elements)

            probs = utils.exp_normalize(priors + log_liks)
            choice = self.rng.choice(len(probs), p=probs)

            self.move_row(row, cluster_i, choice)

    def transition_params(self):
        for dim in self.dims:
            dim.transition_params(self.elements)

    def move_row(self, row, orig, dest):
        """move a row from one cluster to another"""
        added = self._add_row(row, dest)

        if added:
            self._remove_row(row, orig)


    def get_dim_indeces(self):
        return [dim.index for dim in self.dims]

    def _add_row(self, row, cluster_i):
        """add a row to a cluster"""
        if cluster_i == self.K:
            for dim in self.dims:
                dim.params.append(dim.gen_params(1))

        return super().add_obj(row, cluster_i)

    def _remove_row(self, row, cluster_i):
        """remove a row from a cluster"""
        len_before = self.K
        super().remove_obj(row, cluster_i)
        len_after = self.K

        if len_before > len_after:
            for dim in self.dims:
                del dim.params[cluster_i]
