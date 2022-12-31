import numpy as np
from numpy.random import default_rng
from scipy.special import gammaln


class CRP:
    """Basic constructor for classes that need to do Chinese Restaurant Process clustering.
    Assumes that all objects to be clustered are unique and hashable."""

    def __init__(self, alpha):
        """
        Args:
            alpha (int): concentration parameter for the CRP

        Params:
            N (int): number of objects currently seated
            K (int): number of clusters currently
            elements (list(set)): a list of sets, each containing the elements in a cluster
        """
        self.rng = default_rng()

        self.alpha = alpha
        self.N = 0
        self.K = 0
        self.elements = []

    def cluster_of(self, obj):
        """return the cluster index of the object"""
        for index, cluster in enumerate(self.elements):
            if obj in cluster:
                return index

        return None

    def add_obj(self, obj, cluster_i):
        """add an object to a specific cluster"""
        if cluster_i == self.K:
            self.elements.append({obj})
            self.K += 1

        elif obj in self.elements[cluster_i]:
            return 0
        else:
            self.elements[cluster_i].add(obj)

        self.N += 1
        return 1

    def remove_obj(self, obj, cluster_i):
        """remove an object from a specific cluster"""
        try:
            cluster = self.elements[cluster_i]
        except:
            print(f"failed to remove obj {obj} from cluster {cluster_i}")
            return None

        cluster.remove(obj)

        if len(cluster) == 0:
            del self.elements[cluster_i]
            self.K -= 1

        self.N -= 1

    def assign(self, items):
        """assign items to clusters"""
        for item in items:
            self.N += 1

            if self.N == 1:
                self.elements.append({item})
                self.K += 1
            else:
                probs = self.calc_assignment_probs()
                choice = self.rng.choice(len(probs), p=probs)

                if choice == self.K:
                    self.elements.append({item})
                    self.K += 1
                else:
                    self.elements[choice].add(item)

        return self

    def calc_assignment_probs(self):
        """P(x in c_i|alpha,N,K). last element of the array is P(x in new cluster|...)"""
        assignment_probs = [
            (len(cluster) / (self.N - 1 + self.alpha)) for cluster in self.elements
        ]
        assignment_probs.append((self.alpha / (self.N - 1 + self.alpha)))

        return assignment_probs

    def calc_logpdf_marginal(self, alpha):
        """log P(alpha|N,K)"""

        return (
            self.K * np.log(alpha)
            + np.sum(gammaln([len(cluster) for cluster in self.elements]))
            + gammaln(alpha)
            - gammaln(self.N + alpha)
        )

    def calc_transition_log_probs(self, cluster_i):
        """unnormalized log P(x in c_i|alpha,N,K), assuming that x was taken out of cluster i"""
        probs = [len(cluster) for cluster in self.elements]
        probs.append(self.alpha)

        # if cluster is a singleton, CRP prob is of assigning a new cluster
        if len(self.elements[cluster_i]) == 1:
            probs[cluster_i] = self.alpha
        else:
        # CRP prob is as if it weren't there
            probs[cluster_i] = (len(self.elements[cluster_i]) - 1)

        return np.log(np.array(probs))

    def __str__(self):
        return f"{self.elements}\nalpha: {self.alpha}"
