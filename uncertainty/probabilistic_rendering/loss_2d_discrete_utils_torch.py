##########################
#
# This file is part of https://github.com/TRAILab/UncertainShapePose
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import math
import scipy
import time
import torch
import numpy as np


device = torch.device("cpu") # macbook uses mps
cpu_device = torch.device("cpu")
dtype = torch.float64

BACKGROUND_DEPTH = 9.0

# Always call confine_0_1 before any beta and logit-normal functions
def confine_0_1(x):
    eps = 1e-9
    return torch.clip(x, eps, 1 - eps)

# Flipped sigmoid function with k as the slope parameter
def occupancy_from_sdf(sdfs,delta):
    occs = sdfs.clone()
    ones_indices = sdfs < -delta
    zeros_indices = sdfs > delta
    trans_indices = ~(ones_indices | zeros_indices)
    occs[ones_indices] = 1
    occs[zeros_indices] = 0
    occs[trans_indices] = -occs[trans_indices] / (2 * delta) + 0.5
    return occs


# Calculate termination probability (Beta) parameters for voxels
# Given the logit-normal parameters for voxels m = 0, 1, ..., M-1 along the ray
# determine the termination probability parameters for voxels m = 0, 1, ..., M-1, M(passed through)
def termination_probability(occs):
    M = len(occs[0])
    N = len(occs)

    term_prob_first = occs[:,0]

    term_probs = term_prob_first.reshape((N,1))
    
    for i in range(1, M):
        term_prob_i = occs[:,i].reshape((N,1))
        for j in range(0, i):
            term_prob_i = term_prob_i * (1 - occs[:,j]).reshape((N,1))

        term_probs = torch.hstack((term_probs, term_prob_i.reshape((N,1))))

    term_probs_end = torch.prod(1-occs, dim=1).reshape((N,1))
    term_probs = torch.hstack((term_probs, term_probs_end.reshape((N,1))))

    return term_probs



# Calculate the expected value and variance of the rendered depth
def depth_render(depths, probs):
    N = len(depths)

    mask = torch.isnan(probs) | torch.isinf(probs) | (depths >= BACKGROUND_DEPTH-0.1)
    probs[mask] = 0

    normalizer = torch.sum(probs, dim=1).reshape((N,1))
    probs /= normalizer
    probs = torch.nan_to_num(probs)

    E_d = torch.sum(depths * probs, dim=1)

    return E_d



if __name__ == "__main__":

    # Test approximation of the logit-normal distribution

    import matplotlib.pyplot as plt

    # Test occupancy prob

   
    # Test termination probability
    delta = 0.11
    sdfs = torch.tensor([[0.3, 0.2, 0.1, 0.05, 0.0, -0.05, -0.1, -0.2, -0.3],[0.3, 0.2, 0.11, 0.08, 0.0, -0.08, -0.11, -0.2, -0.3]], device=device, dtype=dtype)
    occs = occupancy_from_sdf(sdfs,delta)
    print("Occupancy", occs)

    
    voxel_sdfs = torch.tensor([[0.8,0.6,0.4,0.1,-0.1, -0.3],[0.7,0.5,0.3,0.0,-0.2, -0.3]], device=device, dtype=dtype)
    #voxel_sigmas= torch.tensor([[0.05,0.05,0.05,0.05,0.05,0.05],[0.05,0.05,0.05,0.05,0.05,0.05]], device=device, dtype=dtype)
    voxel_depths = torch.tensor([[1.1,1.2,1.3,1.4,1.5,1.6],[1.1,1.2,1.3,1.4,1.5,1.6]], device=device, dtype=dtype)

    voxel_occs = occupancy_from_sdf(voxel_sdfs,delta)

    time_rt_begin = time.time()
    term_probs = termination_probability(voxel_occs)
    time_rt_end = time.time()
    print("RT runtime duration", time_rt_end - time_rt_begin)

    for n in range(len(voxel_occs)):
        for i in range(len(voxel_occs[0])):
            print("Termination Prob for ray", n, "at index", i, "is", term_probs[n][i])
        print('\n')

    
    E_d = depth_render(voxel_depths, term_probs[:,:-1])
    print("Expected depth", E_d)