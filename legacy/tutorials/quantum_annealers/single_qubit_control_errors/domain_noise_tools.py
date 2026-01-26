# This code is part of QCMet.
# 
# (C) Copyright 2024 National Physical Laboratory and National Quantum Computing Centre 
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np
import scipy.optimize as opt


def single_noisy_sample(beta, field_noise):
    """ A single sample of a domain wall locations on a noisy chain
    
    Args:
    	beta: the inverse temperature given as a number
    	field_noise: a 1-D array of length chain_length (the number of domain wall sites -1) which gives the field values
    """
    E_array = np.zeros(len(field_noise) + 1)  # array for field values
    for i_sites in range(len(E_array)):  # loop through domain wall sites
        E_array[i_sites] = sum(field_noise[:i_sites])  # sum of all previous fields
    E_array = E_array - np.amin(
        E_array)  # change zero of energy to make all non-negative to avoid numerical blowup at low temperature
    p_array = np.exp(-beta * E_array)  # un-normalised probabilities
    Z = sum(p_array)  # sum to find partition function
    p_array = p_array / Z  # normalise probabilities
    return p_array  # return probabilities


def generate_chain_probabilities(beta, noise_ens):
    """Generates probabilites of domain wall locations with inverse temperature beta based on a noise ensemble noise_ens, assumes chain breaks are not possible
    
    Args:
		beta: the inverse temperature given as a number
		noise_ens: an array of size [chain_length,sample_number], where site number is the chain_length (the number of potential domain-wall sites-1), and sample_number is the number of samples to be taken
		
    Returns:
      	An array of length site_number which gives the probability of finding the domain wall on each site
    """
    n_sample = noise_ens.shape[1]  # number of samples used to approximate distribution
    n_site = noise_ens.shape[0] + 1  # number of potential domain wall locations
    p_dist = np.zeros(n_site)  # empty array for probability distribution
    for i_sample in range(n_sample):  # loop through samples
        p_dist = p_dist + single_noisy_sample(beta, noise_ens[:, i_sample]) / n_sample  # add to total probability distribution
    return p_dist  # return the total probability distribution


def calculate_beta_error(beta, data, noise_ens, exclude_endpoints=False):
    """Calculates error from real data for a given beta value, assuming normally distributed noise scaled to stength 1
    
    Args:
		beta: inverse temperature given as a number
		data: site probability data to which the noise is compared
		noise_ens: an array of size [chain_length,sample_number], where site number is the chain_length (the number of potential domain-wall sites-1), and sample_number is the number of samples to be taken
		exclude_enpoints: a flag which escludes the end of the chain from the caluculation if there are effects on terminal sites which may interfere with data
	
    Returns:
      	An array of length site_number which gives the probability of finding the domain wall on each site
    """
    p_dist = generate_chain_probabilities(beta, noise_ens)  # calculate the probability distribution based on noise and temperature
    diff = data - p_dist  # difference in values
    if exclude_endpoints:
        diff = diff[1:-1]  # exclude the ends
    return diff


def least_squares_beta_extract(data, beta0=1, n_sample=100000, exclude_endpoints=False):
    """Use non-linear least squares to extract ratio of noise to temperature
    
    Args:
		data: raw domain wall location data
		beta0: start point of the search defaults to beta=1
		n_sample: number of samples to use in the noise ensemble
		exclude_endpoints: a flag which is passed to the error calculating function
    
    Returns:
    	Both the value of the fitted noise/temperature ratio (beta) and the full least squares fitting data
    """
    n_site = len(data)  # number of sites
    noise_ens = np.random.normal(size=(n_site - 1, n_sample))  # generate a single noise ensemble to prevent statistical flututations from interfering with convergence
    ls_data = opt.least_squares(lambda beta: calculate_beta_error(beta, data, noise_ens, exclude_endpoints=exclude_endpoints), beta0)
    if not ls_data['success']:  # print warning if the fit failed
        print('Least squares fitting has failed, data will still be returned but use caution')
    return [abs(ls_data['x'][0]), ls_data]  # absolute values since negative betas are also valid by symmetry
