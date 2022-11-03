import pymc as pm
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import aesara.tensor as at

def plot_network(network, 
                 pos=None, 
                 ax=None, 
                 title=None, 
                 node_color=None, 
                 node_size=None, 
                 edge_width=0.5):
    if (pos is None and nx.get_node_attributes(network, 'pos') is None) or (pos is None and len(nx.get_node_attributes(network, 'pos'))==0):
        pos = nx.spring_layout(network)
    elif pos is None:
        pos = nx.get_node_attributes(network, 'pos')
    n = len(network.nodes())
    if node_color is None:
        node_color = n*['k']
    clean_ax = True
    if ax is None:
        plt.figure(facecolor='w')
        ax = plt.gca()
        clean_ax = False
    if node_size is None:
        d = dict(nx.degree(network))
        node_size = [v**2/10 for v in d.values()]

    nx.draw_networkx(
        network,
        ax=ax,
        pos=pos,
        node_color=node_color,
        node_size=node_size,
        linewidths=2.0,  # node borders
        edge_color='#333333',
        width=edge_width,  # edge widths
        with_labels=False)
    if title is not None:
        ax.set_title(title, color='k', fontsize='24')
    ax.axis('off')
    if not clean_ax:
        plt.tight_layout()
        plt.show()
    return ax

def logistic_curve(x, x0=0, rate=1, L=1):
    return L / (1 + np.exp(-rate*(x-x0)))

def pymc_logistic_curve(x, x0=0, rate=1, L=1):
    return L / (1 + pm.math.exp(-rate*(x-x0)))

def polar2carthesian(r, phi):
    return np.array([r * np.cos(phi), r * np.sin(phi)]).T

def triu2mat(v, n=None):
    if n is None:
        m = len(v)
        n = int((1 + np.sqrt(1 + 8 * m)) / 2)
    mat = np.zeros((n, n))
    triu_indices = np.triu_indices(n, k=1)
    mat[triu_indices] = v
    return mat + mat.T

def arccosh(x, cap_at_1=False):
    # Note: arccosh(x) = ln(x + sqrt(x ^ 2 - 1))
    # See https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#Inverse_hyperbolic_cosine
    # Numerical precision issues may make the result from arccosh < 1, which is invalid; hence the at.math.largest(1, x)
    if cap_at_1:
        return at.math.largest(1, pm.math.log(x + pm.math.sqrt(pm.math.sqr(x) - 1)))
    return pm.math.log(x + pm.math.sqrt(pm.math.sqr(x) - 1))


def node_pos_dict2array(pos_dict):
    """
    Put the latent positions into a dictionary: {node: position}.
    """
    pos_array = np.zeros((n, D))
    for i in range(n):
        pos_array[i, :] = pos_dict[i]
    return pos_array
