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
from loss_2d_discrete_utils_torch import *

data_type = 'chair'

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

min_ray_sample = 2
sdf_threshold = 0.025

tensor_mus = torch.tensor(mus, device=device, dtype=dtype)
tensor_depths = torch.tensor(depths, device=device, dtype=dtype)

Nu = mus.shape[0]
Nv = mus.shape[1]

sdf_threshold = 0.025
delta = 0.01
n_pts = len(depths)
min_ray_sample = 2
max_ray_sample = 20
N_coords = Nu * Nv

output_depth = np.zeros((mus.shape[0], mus.shape[1]))

pixel_coords = torch.tensor([(u, v) for u in range(Nu) for v in range(Nv)], device=device, dtype=torch.int)

batch_size = 4096
for i in range(0, N_coords, batch_size):
    print("Working on batch", int(i / batch_size)+1,"out of", int(np.ceil(N_coords / batch_size)))

    time_preprocess_start = time.time()

    batch_coords = pixel_coords[i:i+batch_size]

    us, vs = batch_coords[:, 0], batch_coords[:, 1]

    batch_mus = tensor_mus[us, vs]
    batch_depths = tensor_depths.repeat(len(batch_coords), 1)

    mask = torch.abs(batch_mus) <= sdf_threshold

    mask_pos = batch_mus > 0
    mask_neg = batch_mus < 0

    zero_mask_non_zero = torch.any(mask, dim=1)
    zero_mask_suff_pts = torch.sum(mask, dim=1) >= min_ray_sample
    zero_mask = zero_mask_non_zero & zero_mask_suff_pts

    pos_mask = torch.any(mask_pos, dim=1)
    neg_mask = torch.any(mask_neg, dim=1)

    zero_mask = zero_mask_non_zero & zero_mask_suff_pts & pos_mask & neg_mask

    if not torch.any(zero_mask):
        continue

    us = us[zero_mask]
    vs = vs[zero_mask]

    shifted_mask_full, shift_indices_full = (mask.int() * 1).sort(dim=1, descending=True, stable=True)
    shifted_mask = shifted_mask_full[:,:max_ray_sample].bool()
    shift_indices = shift_indices_full[:,:max_ray_sample]

    shifted_mus = (batch_mus.gather(1, shift_indices) * shifted_mask) [zero_mask]
    shifted_depths = (batch_depths.gather(1, shift_indices) * shifted_mask) [zero_mask]

    shifted_mus[~shifted_mask[zero_mask]] = sdf_threshold
    shifted_depths[~shifted_mask[zero_mask]] = BACKGROUND_DEPTH

    time_preprocess_end = time.time()
    print("    Preprocess time:", time_preprocess_end - time_preprocess_start)

    time_render_start = time.time()

    voxel_occs = occupancy_from_sdf(shifted_mus, delta)
    term_probs = termination_probability(voxel_occs)

    E_d = depth_render(shifted_depths, term_probs[:,:-1])
    
    time_render_end = time.time()

    
    print("    Render time:", time_render_end - time_render_start)

    time_postprocess_start = time.time()

    output_depth[us,vs] = E_d.detach().numpy()

    time_postprocess_end = time.time()

    print("    Postprocess time:", time_postprocess_end - time_postprocess_start)

np.save('./output/depth_discrete.npy', output_depth)

scipy.io.savemat('./output/depth_discrete.mat', {'depth': output_depth})

plt.imsave('./output/depth_discrete.png', output_depth)

        
        