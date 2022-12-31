import numpy as np
from numpy.random import default_rng
from scipy.special import betaln
from RunDEMC import Model, Param, dists

import utils


class Dim:
    """a Dim object, associated with a column of the dataframe"""

    def __init__(self, index, data, hypers, params):
        """
        Args:
            index (int): index of the Dim in the dataframe. mostly exists so that Dims can be hashed
            data (np.array): the data associated with that Dim
            hypers (dict): maps hyperparameter names to values <SUBJECT TO CHANGE>
            params (list): list containing the parameter for each row cluster
        """
        self.index = index
        self.data = data
        self.hypers = hypers
        self.params = params

    def gen_params(self, n_clusters):
        """generate cluster parameters from the hyperprior"""
        params = list(
            default_rng().beta(
                a=(self.hypers["s"] * self.hypers["b"]),
                b=(self.hypers["s"] * (1 - self.hypers["b"])),
                size=n_clusters,
            )
        )
        return params

    def calc_beta_sufficient_statistics(self):
        # stop log(0) error, there's almost certainly a better way to do this
        # if the rate parameter is 1
        params_arr = np.array(self.params)

        for i in range(len(self.params)):
            if self.params[i] == 1:
                self.params[i] -= 0.01
            if self.params[i] == 0:
                self.params[i] += 0.01

        if 0 in params_arr or 1 in params_arr:
            print(params_arr)

        sum_log_x = np.sum(np.log(params_arr))
        sum_minus_log_x = np.sum(np.log(1 - params_arr))

        return (sum_log_x, sum_minus_log_x, len(self.params))

    def transition_hypers(self):
        # TODO: to make faster, reparameterize so I don't have to backtransform every iteration
        # the internet says quadrature works very well here

        def beta_log_lik(pop, *args):
            sum_log_x = args[0]
            sum_minus_log_x = args[1]
            K = args[2]

            alpha = pop["s"] * pop["b"]
            beta = pop["s"] * (1.0 - pop["b"])

            log_lik = 0
            log_lik -= K * betaln(alpha, beta)
            log_lik += (beta - 1.0) * sum_minus_log_x
            log_lik += (alpha - 1.0) * sum_log_x

            return log_lik

        params = [
            Param(name="s", prior=dists.exp(5)),
            Param(name="b", prior=dists.beta(1, 1)),
        ]

        mod = Model(
            name="fun",
            params=params,
            like_fun=beta_log_lik,
            like_args=self.calc_beta_sufficient_statistics(),
            initial_zeros_ok=False,
            use_priors=True,
            verbose=False,
        )

        burnin = 200
        mod(400, burnin=False)
        self.hypers = utils.posterior_sample(mod, burnin)
        # faster if I just choose the first sample after the burn-in period
        # is it better or worse if I use the MAP? bias-variance tradeoff...

    def transition_params(self, clusters):
        """beta is the conjugate prior to bernoulli so calculation is closed form"""
        alpha = self.hypers["s"] * self.hypers["b"]
        beta = self.hypers["s"] * (1.0 - self.hypers["b"])

        for i, cluster in enumerate(clusters):
            N = len(cluster)
            K = np.sum(self.data[list(cluster)])
            alpha_prime = alpha + K
            beta_prime = beta + N - K
            self.params[i] = alpha_prime / (alpha_prime + beta_prime)

    def calc_bernpdf_marginal_dim(self, clusters):
        """log marginal likelihood of observing this dim under a clustering scheme
        https://gregorygundersen.com/blog/2020/08/19/bernoulli-beta/"""

        alpha = self.hypers["s"] * self.hypers["b"]
        beta = self.hypers["s"] * (1.0 - self.hypers["b"])
        log_prob = 0

        for i, cluster in enumerate(clusters):
            N = len(cluster)
            K = np.sum(self.data[list(cluster)])
            alpha_prime = alpha + K
            beta_prime = beta + N - K
            log_prob += betaln(alpha_prime, beta_prime) - betaln(alpha, beta)

        return log_prob

    def calc_post_bern_row(self, row, clusters):
        """posterior log predictive of observing a row in every existing cluster for this dimension"""
        
        log_liks = np.zeros(len(clusters) + 1)

        for i, cluster in enumerate(clusters):
            x = self.data[row]
            K = np.sum(self.data[list(cluster)])
            N = len(cluster)
        
            if row in cluster:
                K -= x
                N -= 1
        
            log_liks[i] += self.calc_post_bern(x, N, K)
        
        log_liks[len(clusters)] = self.calc_post_bern(x, 0, 0)

        return log_liks    

    def calc_post_bern(self, x, N, K):
        """posterior predictive of observing x in one cluster"""
        alpha = self.hypers["s"] * self.hypers["b"]
        beta = self.hypers["s"] * (1.0 - self.hypers["b"])
        if x == 1:
            return (np.log(alpha + K) - np.log(alpha + beta + N))
        else:
            return (np.log(K + beta + N) - np.log(alpha + beta + N))

    def __hash__(self):
        return hash(self.index)

    def __str__(self):
        return f"Dim {self.index}"

    # for debugging
    def __repr__(self):
        return f"Dim {self.index}"
