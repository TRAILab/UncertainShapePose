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

import pickle
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from loss_2d_uncertainty_utils_torch import *

data_type = 'car'

ray_file = './data/' + data_type + '.pkl'
data = pickle.load(open(ray_file, 'rb'))

# car
if data_type == 'car':
    ins_id = 0
    step = 10

    mus = data[ins_id][0][step]  # (pixel_u, pixel_v, sample_num)
    sigmas = np.sqrt(data[ins_id][1][step])  # (pixel_u, pixel_v, sample_num)
    depths = data[ins_id][2][step]  # (sample_num)

    k = 400
    max_ray_sample = 20

# chair
elif data_type == 'chair':
    mus = data['mus']  # (pixel_u, pixel_v, sample_num)
    sigmas = np.sqrt(data['sigmas'])  # (pixel_u, pixel_v, sample_num)
    depths = data['depths'] # (sample_num)

    k = 400
    max_ray_sample = 10

elif data_type == 'car2':
    ins_id = 0
    step = 1

    mus = data[ins_id][0][step]  # (pixel_u, pixel_v, sample_num)
    sigmas = np.sqrt(data[ins_id][1][step])  # (pixel_u, pixel_v, sample_num)
    depths = data[ins_id][2][step]  # (sample_num)

    k = 400
    max_ray_sample = 10

sdf_threshold = 0.025

tensor_mus = torch.tensor(mus, device=device, dtype=dtype)
tensor_sigmas = torch.tensor(sigmas, device=device, dtype=dtype)
tensor_depths = torch.tensor(depths, device=device, dtype=dtype)

Nu = mus.shape[0]
Nv = mus.shape[1]
N_coords = Nu * Nv

output_depth = np.zeros((mus.shape[0], mus.shape[1]))
output_std = np.zeros((mus.shape[0], mus.shape[1]))

pixel_coords = torch.tensor([(u, v) for u in range(Nu) for v in range(Nv)], device=device, dtype=torch.int)

batch_size = 4096
for i in range(0, N_coords, batch_size):
    print("Working on batch", int(i / batch_size)+1,"out of", int(np.ceil(N_coords / batch_size)))

    time_preprocess_start = time.time()

    batch_coords = pixel_coords[i:i+batch_size]

    us, vs = batch_coords[:, 0], batch_coords[:, 1]

    batch_mus = tensor_mus[us, vs]
    batch_sigmas = tensor_sigmas[us, vs]
    batch_depths = tensor_depths.repeat(len(batch_coords), 1)

    mask_sdf = torch.abs(batch_mus) <= sdf_threshold

    shifted_mask_full, shift_indices_full = (mask_sdf.int() * 1).sort(dim=1, descending=True, stable=True)

    max_ray_sample_batch = max_ray_sample

    shifted_mask = shifted_mask_full[:,:max_ray_sample_batch].bool()
    shift_indices = shift_indices_full[:,:max_ray_sample_batch]

    shifted_mus = (batch_mus.gather(1, shift_indices) * shifted_mask)
    shifted_sigmas = (batch_sigmas.gather(1, shift_indices) * shifted_mask)
    shifted_depths = (batch_depths.gather(1, shift_indices) * shifted_mask)

    shifted_mus[~shifted_mask] = BACKGROUND_DEPTH
    shifted_sigmas[~shifted_mask] = sdf_threshold / 20
    shifted_depths[~shifted_mask] = BACKGROUND_DEPTH
    shifted_depths = torch.hstack((shifted_depths, BACKGROUND_DEPTH * torch.ones((len(shifted_depths), 1), device=device, dtype=dtype)))

    time_preprocess_end = time.time()
    print("    Preprocess time:", time_preprocess_end - time_preprocess_start)

    time_render_start = time.time()

    term_alphas, term_betas = termination_probability_params_from_logitn(k, shifted_mus, shifted_sigmas)
    E_d, Var_d = depth_render(shifted_depths, term_alphas, term_betas)
    
    time_render_end = time.time()

    print("    Render time:", time_render_end - time_render_start)

    time_postprocess_start = time.time()

    output_depth[us,vs] = E_d.detach().numpy()
    output_std[us,vs] = np.sqrt(Var_d.detach().numpy())

    time_postprocess_end = time.time()

    print("    Postprocess time:", time_postprocess_end - time_postprocess_start)

np.save('./output/depth.npy', output_depth)
np.save('./output/std.npy', output_std)

scipy.io.savemat('./output/depth.mat', {'depth': output_depth})
scipy.io.savemat('./output/std.mat', {'std': output_std})

plt.imsave('./output/depth.png', output_depth)
plt.imsave('./output/std.png', output_std)

        
        