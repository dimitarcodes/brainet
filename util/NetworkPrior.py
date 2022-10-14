from util.NetworkLikelihood import *
from util.NetworkModel import *
from util.util import *

class NetworkPrior:

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

    def __init__(self, n, name='Stochastic Block'):
        super().__init__(n, name)
    
    def distribution(self):
        raise NotImplementedError

class ErdosRenyi(NetworkPrior):
    
    def __init__(self, n, a=1, b=1, name='Erdös-Renyí'):
        self.a, self.b = a, b
        super().__init__(n, name)


    #
    def distribution(self):
        p_scalar = pm.Beta('p_scalar', alpha=self.a, beta=self.b)
        p = pm.Deterministic('p', p_scalar * at.ones((self.n, self.n))[self.triu_indices])
        return p


    #
#


class ExtendedErdosRenyi(NetworkPrior):

    def __init__(self, n, a=1, b=1, name='Extended Erdös-Renyí'):
        self.a, self.b = a, b
        super().__init__(n, name)


    #
    def distribution(self):
        n, a, b, triu_indices = self.n, self.a, self.b, self.triu_indices
        p_i = pm.Beta('p_i', alpha=a, beta=b, shape=(n, 1))
        p = pm.Deterministic('p', (at.repeat(p_i, axis=1, repeats=n) * \
                                   at.transpose(at.repeat(p_i, axis=1, repeats=n)))[triu_indices])
        return p


    #
#
class EuclideanLatentSpace(NetworkPrior):

    def __init__(self, n, D, x0=0.0, T=1, name='Euclidean LSM'):
        self.D, self.x0, self.T = D, x0, T
        super().__init__(n, name)

    
    #
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
    

    #
#


class HyperbolicLatentSpace(NetworkPrior):

    def __init__(self, n, R=1, alpha=1, x0=0.0, T=1, name='Hyperbolic LSM'):
        self.R, self.x0, self.T, self.alpha = R, x0, T, alpha
        super().__init__(n, name)

    
    #
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
    

    #
#