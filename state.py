import numpy as np
from numpy.random import default_rng
import scipy.stats as stats

import utils
from crp import CRP
from dim import Dim
from view import View


class State(CRP):
    def __init__(self, data, alpha, rng=None):
        """Initialize a state by sampling it from the priors"""
        super().__init__(alpha)
        self.rng = rng

        if self.rng is None:
            self.rng = default_rng()

        self.n_obs, self.n_dims = data.shape
        self.data = data

        # create dims, dim hypers
        self.dims = []
        for dim_i in range(self.n_dims):
            hypers = {
                "s": self.rng.exponential(5),
                "b": self.rng.beta(a=1, b=1),
            }

            dim = Dim(index=dim_i, data=data[:, dim_i], hypers=hypers, params=None)
            self.assign([dim])
            self.dims.append(dim)

        # create views, create dim parameters
        self.views = []
        for dims in self.elements:
            view = View(dims=dims, alpha=self.gen_alpha())
            view.assign(range(self.n_obs))
            self.views.append(view)

            for dim in view.dims:
                dim.params = dim.gen_params(view.K)

    def gen_alpha(self):
        """generate CRP concentration param from prior"""
        return self.rng.gamma(1, 1)

    def transition(self, sweeps=10):
        for i in range(sweeps):
            print(f"sweep {i+1}/{sweeps}")
            self.transition_alphas()
            self.transition_hypers()
            self.transition_clusters()
            self.transition_dims()
            self.transition_params()

    def transition_alphas(self):
        """Currently grid approximation, very fast + explicitly recommended by paper for this section"""

        # discrete approximation of P(alpha_d|views)
        grid = utils.log_linspace(1.0 / self.n_dims, self.n_dims, 100)
        log_lik = np.array([self.calc_logpdf_marginal(x) for x in grid])
        # TODO: factor hyperprior in here?
        self.alpha = self.rng.choice(grid, p=utils.exp_normalize(log_lik))

        # discrete approximation of P(alpha_v_i|clusters)
        grid = utils.log_linspace(1.0 / self.n_obs, self.n_obs, 100)
        for view in self.views:
            log_lik = np.array([view.calc_logpdf_marginal(x) for x in grid])
            # TODO: factor hyperprior in here?
            view.alpha = self.rng.choice(grid, p=utils.exp_normalize(log_lik))

    def transition_hypers(self):
        """transition cluster-parameter generating hyperparameters for each dimension"""
        for dim in self.dims:
            dim.calc_beta_sufficient_statistics()
            dim.transition_hypers()

    def transition_clusters(self):
        for view in self.views:
            view.transition_clusters()

    def transition_dims(self):
        for dim in self.dims:  # TODO: does this have to be random?
            view_i = self.cluster_of(dim)
            # self._verify_views() # for testing

            # calc priors
            priors = self.calc_transition_log_probs(view_i)

            # calc likes:
            new_view = View(self.gen_alpha(), {dim})
            new_view.assign(range(self.n_obs))
            log_liks = [
                dim.calc_bernpdf_marginal_dim(view.elements) for view in self.views
            ]
            log_liks.append(dim.calc_bernpdf_marginal_dim(new_view.elements))

            probs = utils.exp_normalize(priors + np.array(log_liks))
            choice = self.rng.choice(len(probs), p=probs)

            # dims didn't change spots
            if choice == view_i:
                continue

            # dim belongs to a brand new category
            if choice == self.K:
                self.add_obj(dim, choice)

                dim.params = dim.gen_params(new_view.K)
                new_view.dims = self.elements[choice]
                # I want a reference to the existing, not a new set
                self.views.append(new_view)

                self.remove_obj(dim, view_i)

            # dim belongs to an existing category
            else:
                self.add_obj(dim, choice)

                dim.params = dim.gen_params(self.views[choice].K)

                self.remove_obj(dim, view_i)

            # remove the old view from the views list if I deleted a view
            if len(self.views[view_i].dims) == 0:
                del self.views[view_i]

    def transition_params(self):
        for view in self.views:
            view.transition_params()

    def calc_log_lik_model(self):
        """return the log likelihood of the model"""
        log_lik = np.longfloat(0)
        
        # prior prob of observing each view
        log_lik += self.calc_logpdf_marginal(self.alpha)
        log_lik -= self.alpha 

        # likelihood of observing the hyperpriors
        for dim in self.dims:
            lik_s = stats.expon.logpdf(x=dim.hypers["s"], loc=1/5)
            lik_b = stats.beta.logpdf(x=dim.hypers["b"], a=1, b=1)
            log_lik += lik_s + lik_b

        # prior prob of observing each cluster
        for view in self.views:
            log_lik += view.calc_logpdf_marginal(view.alpha)
            log_lik -= view.alpha
        
        # joint likelihood of observing dims in the views, clusters they are in
        for view in self.views:
            for dim in view.dims:         
                log_lik += dim.calc_bernpdf_marginal_dim(view.elements)
        
        return log_lik


    def _verify_views(self):
        for view, dims in zip(self.views, self.elements):
            assert view.dims == dims, f"objects {view.dims}, {dims} should be the same"

        assert len(self.views) == len(
            self.elements
        ), f"views crp {len(self.elements)} and list {len(self.views)} should be the same length"
