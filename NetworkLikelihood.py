from NetworkModel import *
from NetworkPrior import *
from util import *

import pymc as pm

class NetworkLikelihood:
    def __init__(self, name='Unknown likelihood'):
        self.name = name
    
    def distribution(self):
        raise NotImplementedError

class SingleNetworkObservation(NetworkLikelihood):
    ''' Likelihood function 
    
    '''

    def __init__(self, name='SingleNetworkObservation'):
        super().__init__(name)

    def distribution(self, model, obs=None, m=None):
        if obs is None:
            assert m is not None, 'must specify the number of edges'
            A = pm.Bernoulli('A', p=model['p'], shape=m)  
        else:
            m = len(obs)
            A = pm.Bernoulli('A', p=model['p'], observed=obs, shape=m)
        return A

class MultiNetworkObservation(NetworkLikelihood):

    def __init__(self, num_instances, name='MultiNetworkObservation'):
        self.num_instances = num_instances
        super().__init__(name)

    def distribution(self, model, obs=None, m=None):
        if obs is None:
            assert m is not None, 'must specify the number of edges'
            A = [ pm.Bernoulli('A_{:d}'.format(j), p=model['p'], shape=m) for j in range(self.num_instances) ]
        else:
            m = len(obs[0])
            A = [ pm.Bernoulli('A_{:d}'.format(j), p=model['p'], shape=m, observed=obs[j]) for j in range(self.num_instances) ]
        return A

