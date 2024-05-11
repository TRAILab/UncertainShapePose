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


import time
import numpy as np
import matplotlib.pyplot as plt
from beta_dist import *

data = np.load('./ray_distributions_s1000.npy')

step = 20
ray = data[0,:,step,:]
mus = ray[0,:]
sigmas = np.sqrt(ray[1,:])

sdf_threshold = 0.01

valid_voxels = np.where(np.abs(mus) < sdf_threshold)[0]
voxel_mus = mus[valid_voxels]
voxel_sigmas = sigmas[valid_voxels]

print("SDF means:", voxel_mus)
print("SDF sigmas:", voxel_sigmas)

k = 400
xs = np.linspace(0.0, 1.0, 1000)

plt.figure(1)
for i in range(len(voxel_mus)):
    plt.plot(xs, beta_pdf(xs,*beta_param_estimator(*logit_normal_mean_and_variance(k, voxel_mus[i],voxel_sigmas[i]))))
plt.savefig('rt_tester_1.png')

time_rt_begin = time.time()
term_alphas, term_betas = termination_probability_params_from_logitn(k,voxel_mus,voxel_sigmas)
time_rt_end = time.time()
print("RT runtime duration", time_rt_end - time_rt_begin)
legends = []
plt.figure(2)
for i in range(len(term_alphas)):
    if (i < len(term_alphas)-1):
        print("Termination probability at", i, "with SDF", voxel_mus[i], "is", beta_mean(term_alphas[i],term_betas[i]))
    else:
        print("Probability of passing through is", beta_mean(term_alphas[i],term_betas[i]))
    
    plt.plot(xs, beta_pdf(xs,term_alphas[i],term_betas[i]))
    legends.append("Beta " + str(i+1))

#plt.legend(legends)

# plt.show()
plt.savefig('rt_tester_2.png')
