from brainet.NetworkLikelihood import *
from brainet.NetworkPrior import *
from brainet.util import *


import networkx as nx
import numpy as np

class NetworkModel:
    ''' A class that contains the network model

    
    Attributes
    ----------
    prior : NetworkPrior
        The network prior that the model will use. By default it uses the ErdosRenyi prior.
    
    
    Methods
    -------
    inference(obs, num_draws=500, num_chains=2, num_tune=1000)
        performs inference to update the network model given the observation obs

    sample_from_prior(num_samples=1, num_instances=1)
        generates network samples using the model's network prior.
    '''

    def __init__(self, prior=ErdosRenyi):
        self.prior = prior
        self.model = pm.Model()
        with self.model:
            self.prior.distribution()


    def inference(self, obs, num_draws=500, num_chains=2, num_tune=1000, **kwargs):    
        if isinstance(obs, list):
            self.likelihood = MultiNetworkObservation(num_instances=len(obs))   
        else: 
            self.likelihood = SingleNetworkObservation()
        if isinstance(obs, nx.classes.graph.Graph):
            obs = nx.to_numpy_array(obs)
        
        n = obs.shape[0]
        triu_indices = np.triu_indices_from(obs, k=1)
        adj = obs[triu_indices]
        m = int(n*(n-1)/2)
        
        # add likelihood
        with self.model as model:
            A_obs = pm.Bernoulli('A_obs', p=model['p'], observed=adj, shape=m)
            A = self.likelihood.distribution(model, adj)
            trace = pm.sample(draws=num_draws, 
                              tune=num_tune, 
                              cores=num_chains, 
                              chains=num_chains,
                              **kwargs)
            
        return trace


    #
    def sample_from_prior(self, num_samples=1, num_instances=1):
        '''
        '''
        n = self.prior.n
        m = int(n*(n-1)/2)   # maximum number of edges in an undirected network of size n

        with self.model as model:
            # Note that this assumes all priors have a parameter p of shape m!
            if num_instances > 1:
                self.likelihood = MultiNetworkObservation(num_instances=num_instances)
                A = self.likelihood.distribution(model, m=m)
                # A = [ pm.Bernoulli('A_{:d}'.format(j), p=model['p'], shape=m) for j in range(num_instances) ]
            else:    
                self.likelihood = SingleNetworkObservation()
                A = self.likelihood.distribution(model, m=m)
                # A = pm.Bernoulli('A', p=model['p'], shape=m)  

            # Depending on the model we sample different latents, and of course observations A. 
            samples = pm.sample_prior_predictive(samples=num_samples)

        sampled_networks = list()
        for i in range(num_samples):
            # transform predicted A into a networkx object
            if num_instances > 1:
                networks_i = list()
                for j in range(num_instances):
                    A_ij = triu2mat(samples.prior['A_{:d}'.format(j)][0, i, :])
                    networks_i.append(nx.from_numpy_matrix(A_ij))
            else:
                A_i = triu2mat(samples.prior['A'][0, i, :])
                network_i = nx.from_numpy_matrix(A_i)

            # for some priors, the latents define a position:
            if self.prior.name == 'Euclidean LSM':
                z_i = np.asarray(samples.prior['z'][0, i, :]).squeeze()
            elif self.prior.name == 'Hyperbolic LSM':
                r_i = np.asarray(samples.prior['r'][0, i, :]).squeeze()
                phi_i = np.asarray(samples.prior['phi'][0, i, :]).squeeze()    
                z_i = polar2carthesian(r_i, phi_i)

            if self.prior.name in ['Euclidean LSM', 'Hyperbolic LSM']:
                node_attr_dict_i = {i: list(z_i[i, :]) for i in range(n)}
                if num_instances > 1:
                    for j in range(num_instances):
                        nx.set_node_attributes(networks_i[j], node_attr_dict_i, 'pos')
                else:
                    nx.set_node_attributes(network_i, node_attr_dict_i, 'pos')

            if num_instances > 1:
                sampled_networks.append(networks_i)
            else:
                sampled_networks.append(network_i)
        
        if num_samples == 1:
            return sampled_networks[0]    
        return sampled_networks

    #