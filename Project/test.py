import clips
import random
import numpy as np
#from utils_grad import *
from project_utils_rev1 import *

def generate_bayes_nets(nrows,ncols,storm_prob,sandy_prob):

    T,F = True,False
    
    total_cells = nrows * ncols
    sandy_prob = 3 / total_cells  # Probability of a cell being sandy
    storm_prob = [0.9, 0.25, 0.01]  # Probability of a storm 

    bayesnets = {}

    for i in range(nrows):
        for j in range(ncols):
            # Conditional probability table (CPT) for 'PPM'
            # The CPT is based on the table provided
            ppm_cpt = {
                ('Adjacent', True): {'High': 0.95, 'Medium': 0.04, 'Low': 0.01},
                ('Adjacent', False): {'High': 0.9, 'Medium': 0.09, 'Low': 0.01},
                ('Diagonal', True): {'High': 0.3, 'Medium': 0.5, 'Low': 0.2},
                ('Diagonal', False): {'High': 0.25, 'Medium': 0.5, 'Low': 0.25},
                ('NotNearby', True): {'High': 0.01, 'Medium': 0.24, 'Low': 0.75},
                ('NotNearby', False): {'High': 0.01, 'Medium': 0.09, 'Low': 0.9},
            }

            # Node specifications for the BayesNetCategorical object
            node_specs = [
                ('sandy', '', sandy_prob, [True, False]),
                ('StormProximity', '', storm_prob, ['Adjacent', 'Diagonal', 'NotNearby']),
                ('air_quality', 'StormProximity sandy', ppm_cpt, ['High', 'Medium', 'Low'])
            ]
            #print('NODENODNEONDOENOD',node_specs)

            # Create the BayesNetCategorical object
            cellbn = BayesNetCategorical(node_specs)

            # Add the Bayesian network to the dictionary
            bayesnets[(i, j)] = cellbn

    return bayesnets