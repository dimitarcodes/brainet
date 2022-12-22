from NetworkLikelihood import *
from NetworkModel import *
from util import *

import pymc as pm
import networkx as nx
import numpy as np

import sys

class NetworkPrior:
    """
    An abstract class used to define the prior model of a network.
    Not intended to be used directly.

    Methods
    -------
    distribution() : returns an array specifying the probability of each edge
    occuring

    """

    def __init__(self, n, name='Unknown prior'):
        self.n = n
        self.m = int(n*(n-1)/2)
        self.triu_indices = np.triu_indices(n, k=1)
        self.name = name

    #
    def distribution(self):
        raise NotImplementedError

    #
#

class StochasticBlock(NetworkPrior):
    """ Stochastic Block Model prior
    
    k ~ Poisson(mu)  - Number of blocks/clusters, default mu=1
    Theta ~ Dirichlet(alpha)        - Probability of a node being in cluster K,
                                        alpha=ones(k)
    z_i | Theta ~ Categorical(Theta)- Cluster label of each node 
                                        used Multinomial for one-hot encoding
    pi_between | B1,B2 ~ Beta(B1,B2)- Probability of a connection between:
                                    node i in cluster a
                                    node j in cluster b
                                    a != b
                                        default B1=B2=1
    pi_within | B1,B2 ~ Beta(B1,B2)  - Probability of a connection between:
                                    node i in cluster a
                                    node j in cluster b
                                    a == b
                                        default B1=B2=1
    pi = full class connectivity probability map
    p = z @ pi @ z.T 
    A_i,j | rho, z ~ Bernoulli(p)   the adjacency matrix
    ...

    Attributes
    ----------
    n : int
        number of vertices in the network
    k : int
        number of clusters, if not provided then k is sampled from
        a Poisson(mu) distribution
    mu : int
        number of expected clusters, used in the Poisson(mu) distribution
        used to sample k (if k is provided then mu isn't used)
    between_class : tuple(int int)
        the alpha and beta shape parameters for the Beta distribution used
        to sample the probability of nodes from two different clusters being
        connected
    within_class : tuple(int int)
        the alpha and beta shape parameters for the Beta distribution used
        to sample the probability of nodes from the same cluster being
        connected
    name : str
        the name of the prior
    """

    def __init__(self, n, k=None, mu=4, between_class=(1,1), within_class=(1,1), name='Stochastic Block'):
        '''
        n = number of points to generate
        k = number of clusters
        '''
        super().__init__(n, name)
        self.between_alpha, self.between_beta = between_class
        self.within_alpha, self.within_beta = within_class
        self.mu = mu

        if k is None:
            self.k = lambda : pm.Poisson('k', mu=self.mu)
        else:
            self.k = lambda : pm.DiracDelta('k', k)
            #pm.Constant('k', k)    # deprecated, works but says to use DiracDelta
            #pm.Deterministic('k', aes.shared(k) )  # works but throws warnings
    
    def distribution(self):
        k = self.k()
        
        k_triu_len = (k*(k-1)/2)
        k_triu_indices = at.triu_indices(k, k=1)

        # probability of a node belonging to each cluster (each run has only one array theta)
        theta = pm.Dirichlet('theta', a=at.ones(k)) # Dirichlet(a_1 = a, a_2 = a...) with a=1
        # label of each node, drawn from theta
        z = pm.Multinomial('z', n=1, p=theta, shape=(self.n, k)) # Multinomial with n=1 is just Categorical

        # between classes
        pi_between = pm.Beta('pi_between', alpha=self.between_alpha, beta=self.between_beta, shape=k_triu_len)
        # within classes
        pi_within = pm.Beta('pi_within', alpha=self.within_alpha, beta=self.within_beta, shape=k)

        # reconstruct the full pi matrix
        pi_zeros = at.zeros((k,k))
        pi = at.set_subtensor(pi_zeros[k_triu_indices], pi_between)
        pi = pm.Deterministic('pi', pi + pi.transpose() + at.eye(k)*pi_within)

        # get the full p matrix
        p_full = pm.Deterministic('p_full', z @ pi @ z.transpose())
        p = pm.Deterministic('p', p_full[self.triu_indices])
        return p


class ErdosRenyi(NetworkPrior):
    """ Erdos Renyi network prior model
    
    All edges have the same probability of occuring. This probability is drawn
    from a Beta distribution (default is Beta(1,1)). Whether a particular edge
    is present is then determined by performing a Bernoulli(p) flip.

    ...

    Attributes
    ----------
    n : int
        number of vertices in the network
    a : float
        alpha shape parameter for Beta distribution from which edge probability is drawn
        default value is 1
    b : float
        beta shape parameter for Beta distribution from which edge probability is drawn
        default value is 1

    """
    
    def __init__(self, n, a=1, b=1, name='Erdös-Renyí'):
        self.a, self.b = a, b
        super().__init__(n, name)

    def distribution(self):
        # draw the edge probability from a beta(1,1) distribution
        p_scalar = pm.Beta('p_scalar', alpha=self.a, beta=self.b)
        p = pm.Deterministic('p', p_scalar * at.ones((self.n, self.n))[self.triu_indices])
        return p

class ExtendedErdosRenyi(NetworkPrior):
    """ Extended Erdos Renyi network prior model

    The probability of each edge is drawn from a Beta distribution 
    (default is Beta(1,1)). Whether a particular edge
    is present is then determined by performing a Bernoulli(p) flip.

    ...

    Attributes
    ----------
    n : int
        number of vertices in the network
    a : float
        alpha shape parameter for Beta distribution from which edge probability is drawn
    b : float
        beta shape parameter for Beta distribution from which edge probability is drawn


    """
    def __init__(self, n, a=1, b=1, name='Extended Erdös-Renyí'):
        self.a, self.b = a, b
        super().__init__(n, name)


    def distribution(self):
        n, a, b, triu_indices = self.n, self.a, self.b, self.triu_indices
        p_i = pm.Beta('p_i', alpha=a, beta=b, shape=(n, 1))
        p = pm.Deterministic('p', (at.repeat(p_i, axis=1, repeats=n) * \
                            at.transpose(at.repeat(p_i, axis=1, repeats=n)))[triu_indices])
        return p

class EuclideanLatentSpace(NetworkPrior):
    """ Euclidian Latent Space network model prior
    

    ...

    Attributes
    ----------
    n : int
        number of vertices in the network
    D : 
    x0 : float
    T : int

    """
    def __init__(self, n, D, x0=0.0, T=1, name='Euclidean LSM'):
        self.D, self.x0, self.T = D, x0, T
        super().__init__(n, name)

    
    def distribution(self):
        """
        The Euclidean latent space prior.
        """
        epsilon = 1e-16
        n, D, T, x0, triu_indices = self.n, self.D, self.T, self.x0, self.triu_indices        
        z = pm.Normal('z', shape=(n, D), mu=0., sigma=1.)  # Is this model way faster with uniform coordinates compared to Gaussian?

        # Deterministic transformations
        # Compute Euclidean distance between latent positions, https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
        G = pm.math.dot(z, z.T)  # Gramian matrix zz^T
        G_diag = at.tile(at.nlinalg.extract_diag(G).reshape((n, 1)), n)  # diag(G) 1^T
        distances = pm.Deterministic('D', 
                                    at.math.sqrt(epsilon + at.math.largest(G_diag + G_diag.transpose() - 2 * G, 0)))
        # Numerical precision may result in negative distances, hence max(0, sqrt(epsilon + D^2)).
        # The maxing trick may still result in some infinite gradients, so add epsilon, and adjust num_tune

        # inv-logit transform distances and connection tendency to probabilities
        p = pm.Deterministic('p', pymc_logistic_curve(x=distances[self.triu_indices], 
                                                            x0=x0, 
                                                            rate=-1/T))
        return p
    
class HyperbolicLatentSpace(NetworkPrior):
    """ Hyperbolic Latent Space network model prior
    

    ...

    Attributes
    ----------
    n : int
        number of vertices in the network
    R : 
    alpha1 : 
    x0 : int
    T : 
    """
    def __init__(self, n, R=1, alpha=1, x0=0.0, T=1, name='Hyperbolic LSM'):
        self.R, self.x0, self.T, self.alpha = R, x0, T, alpha
        super().__init__(n, name)

    def distribution(self):
        """
        The hyperbolic latent space prior.
        """
        n, R, alpha, T, x0, triu_indices = self.n, self.R, self.alpha, self.T, self.x0, self.triu_indices
        # Prior on r_i, see Aldecoa et al., 2015:
        U = pm.Uniform('U', lower=0.0, upper=1.0, shape=(n, 1))
        r = pm.Deterministic('r', 1 / alpha * arccosh(1 + (pm.math.cosh(alpha * R) - 1) * U)) 
        # Prior on the angle phi_i:
        phi = pm.Uniform('phi', shape=(n, 1), lower=0., upper=2 * np.pi)

        # Compute hyperbolic distance between latent coordinates
        phi_mat = at.tile(phi, n)
        if 'linux' in sys.platform:
            delta_phi = np.pi - pm.math.abs_(np.pi - pm.math.abs_(phi_mat - phi_mat.transpose()))
        else:
            delta_phi = np.pi - pm.math.abs(np.pi - pm.math.abs(phi_mat - phi_mat.transpose()))
        r_mat = at.tile(r, n)
        r_mat_T = r_mat.transpose()
        x = pm.math.cosh(r_mat) * pm.math.cosh(r_mat_T) - pm.math.sinh(r_mat) * pm.math.sinh(r_mat_T) * pm.math.cos(
            delta_phi)
        
        distances = pm.Deterministic('D', at.fill_diagonal(arccosh(x), 0))  
        p = pm.Deterministic('p', pymc_logistic_curve(x=distances[triu_indices], 
                                                            x0=R, 
                                                            rate=-1/T))
        return p