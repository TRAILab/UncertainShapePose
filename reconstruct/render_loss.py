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

import torch
import numpy as np
from reconstruct.loss import compute_render_loss_nojac, compute_render_depth, sample_with_reparameterization

from reconstruct.loss_utils import get_batch_sdf_values, sdf_to_occupancy

import utils.probability.beta_dist as beta_dist
import os 

def generate_sample_points(K, width, length, t_obj_cam, num_depth_samples=50, uv=None, mask=None, device='cuda'):
    '''
    @t_obj_cam: torch.Tensor
    @uv: (u,v) if specified, only sample one pixel instead of the whole image
    @mask: (width, length) if specified, only sample the pixels round the bbox defined by mask
    '''
    if uv is None:

        if mask is None:
            # sample whole image pixels
            xs = torch.arange(length)
            ys = torch.arange(width)
        else:
            # sample pixels round the mask
            ys_inds, xs_inds = np.where(mask) # x, u; y, v; length, width; -> Storage: (width, length)
            bbox = torch.Tensor([xs_inds.min(), xs_inds.max(), ys_inds.min(), ys_inds.max()]).long()
            xs = torch.arange(bbox[0], bbox[1])
            ys = torch.arange(bbox[2], bbox[3])

        uvs = torch.meshgrid(xs, ys) # generate a whole image pixels
        uvs = torch.stack(uvs).permute(1,2,0).to(device) # (268,388,2)   length, width
    else:
        # only consider one pixel
        uvs = torch.Tensor(uv).long()[None, None, :].to(device)  # (1, 1, 2)

    uvs_homo = torch.cat([uvs, torch.ones(uvs.shape[0],uvs.shape[1], 1).to(device)], dim=-1) # make sure to have first length, and then weigth!
    uvs_homo_flat = uvs_homo.flatten(0,1)
    K_inv = torch.Tensor(K).inverse()
    # (n, 3) = (n, 1, 3) * (3, 3)
    ray_directions = (uvs_homo_flat[:, None, :] * K_inv).sum(-1)  # an origin version. TODO: understand it
    # ray_directions = (uvs_homo_flat[:, None, :] @ K_inv).squeeze(1) # An intutive version
    
    # sampled_ray_depth
    t_cam_obj = torch.inverse(t_obj_cam)

    scale = torch.det(t_cam_obj[:3, :3]) ** (1 / 3)
    depth_min, depth_max = t_cam_obj[2, 3] - 1.0 * scale, t_cam_obj[2, 3] + 1.0 * scale
    sampled_ray_depth = torch.linspace(depth_min, depth_max, num_depth_samples).to(device)

    # (n_rays, num_samples_per_ray, 3) = (n_rays, 1, 3) * (num_samples_per_ray, 1)
    ray_directions = ray_directions
    sampled_points_cam = ray_directions[..., None, :] * sampled_ray_depth[:, None]

    # transform to object coordinates
    # (n_rays, num_samples_per_ray, 3)
    t_obj_cam = torch.inverse(t_cam_obj)

    sampled_points_obj = \
        (sampled_points_cam[..., None, :] * t_obj_cam[:3, :3]).sum(-1) + t_obj_cam[:3, 3]
    
    output = {
        'sampled_points_obj': sampled_points_obj,
        'sampled_ray_depth': sampled_ray_depth
    }
    return output


def decode_uncertainty_sdf(decoder, latent_vector_distribution, query_points_obj, sample_num=10,
                           device='cuda', keep_grad=True):
    '''
    @ latent_vector_distribution: (mean, sigma)
    '''

    # sample with reparameters tricks
    latent_codes_samples = sample_with_reparameterization(latent_vector_distribution, sample_num, logvar_to_sigma=False)

    # (sample_num, points_num, 1)
    # latent_codes_samples = latent_codes_samples.to(device)

    # make sure the inputs are first stored on CPU, then load into GPU one by one
    sdf_batch = get_batch_sdf_values(decoder, latent_codes_samples, query_points_obj, keep_grad=keep_grad)

    # calculate sample mean and sample var
    # (points_num, 1)
    sdf_mean = sdf_batch.mean(0)
    sdf_var = sdf_batch.var(0)

    return sdf_mean, sdf_var

# Debug function
def draw_sdf_sigma_distribution(sdf_mean, sdf_sigma):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(sdf_mean.flatten(), bins=100)
    plt.title('sdf mean histogram')
    plt.savefig('ray_sdf_mean_histogram.png')

    plt.figure()
    plt.hist(sdf_sigma.flatten(), bins=100)
    plt.title('sdf sigma histogram')
    plt.savefig('ray_sdf_sigma_histogram.png')

def compute_depth_distribution(sdf_values, sdf_var, th, sampled_depths):
    '''
    @ sdf_values: (n_samples_per_ray, )
    @ sdf_var: (n_samples_per_ray, )
    @ th: threshold from sdf to occupancy
    '''

    # along the ray, we have n_pts sampled points.
    # each point has its own sdf distribution.
    n_pts = sdf_values.shape[0]

    '''
    # Step 1: calculate the occupancy probability after passing the sdfs
    '''
    # calculate occupancy prob with sigmoid function
    k = 10
    # Num sample for Qausi-Monte Carlo approximation, must be power of 2
    L = 2**4


    sdf_mean = sdf_values
    sdf_sigma = torch.sqrt(sdf_var)
    logit_normal_mean, logit_normal_variance = beta_dist.logit_normal_mean_and_variance_mc(L, k, sdf_mean, sdf_sigma)

    # debug: draw the distribution of the mean/var of this ray.
    draw_sdf_sigma_distribution(sdf_mean, sdf_sigma)

    # get alphas, betas
    mus = logit_normal_mean
    sigmas = logit_normal_variance
    alphas = []
    betas = []
    for i in range(len(mus)):
        alpha_i, beta_i = beta_dist.beta_param_estimator(*beta_dist.logit_normal_mean_and_variance_mc(L, k, mus[i], sigmas[i]))
        alphas.append(alpha_i)
        betas.append(beta_i)


    '''
    # Step 2: calculate the termination probability, by the product of betas
    '''
    # get the product of multiple beta distribution, by taking samples.

    # need to implement the product (consider 1-o, o)
    term_alphas, term_betas = beta_dist.termination_probability_params_1(L, alphas, betas)

    # get final depth distribution
    depth_mean, depth_var = beta_dist.depth_render(sampled_depths, term_alphas, term_betas)

    print('done rendering with beta.')

def get_uv_ray_distribution(sdf_values, sdf_var,
                            width, length,
                            valid_indices, num_depth_samples, mask=None):
    '''
    @ mask: (width, length)
    @ return:
        (length, width, num_depth_samples)
    '''

    max_sdf_value = 1.0
    max_sdf_value_var = 1.0  

    if mask is None:
        sdf_values_uv = torch.full((length * width, num_depth_samples, 1), max_sdf_value)  # for those w/o observations, init as largest
        sdf_var_uv = torch.full((length * width, num_depth_samples, 1), max_sdf_value_var)
        
        sdf_values_uv[valid_indices[0], valid_indices[1], :] = sdf_values.detach().cpu()
        sdf_var_uv[valid_indices[0], valid_indices[1], :] = sdf_var.detach().cpu()

        sdf_values_uv = sdf_values_uv.reshape((length, width, num_depth_samples))
        sdf_var_uv = sdf_var_uv.reshape((length, width, num_depth_samples))
    else:
        # get the depth inside bbox, and then map to the full image
        bbox_inds = np.where(mask) # x, u; y, v; length, width; -> Storage: (width, length)
        bbox = np.array([[bbox_inds[0].min(), bbox_inds[0].max()], [bbox_inds[1].min(), bbox_inds[1].max()]])

        bbox_width = bbox[0, 1] - bbox[0, 0] + 1
        bbox_length = bbox[1, 1] - bbox[1, 0] + 1

        sdf_values_uv_bbox = torch.full((bbox_width * bbox_length, num_depth_samples, 1), max_sdf_value)
        sdf_var_uv_bbox = torch.full((bbox_width * bbox_length, num_depth_samples, 1), max_sdf_value_var)

        # valid_indices: valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)
        # shape of sampled_points_obj: 103984, 50, 3 
        sdf_values_uv_bbox[valid_indices[0], valid_indices[1], :] = sdf_values.detach().cpu()
        sdf_var_uv_bbox[valid_indices[0], valid_indices[1], :] = sdf_var.detach().cpu()

        # use first length, and then width; because for uv to generate rays, its first x, then y.
        sdf_values_uv_bbox_reshaped = sdf_values_uv_bbox.reshape((bbox_length, bbox_width, num_depth_samples))
        sdf_var_uv_bbox_reshaped = sdf_var_uv_bbox.reshape((bbox_length, bbox_width, num_depth_samples))

        # map it to a full image
        sdf_values_uv = torch.full((length, width, num_depth_samples), max_sdf_value) 
        sdf_var_uv = torch.full((length, width, num_depth_samples), max_sdf_value_var)
        sdf_values_uv[bbox[1, 0]:(bbox[1, 1] + 1), bbox[0, 0]:(bbox[0, 1] + 1), :] = sdf_values_uv_bbox_reshaped
        sdf_var_uv[bbox[1, 0]:bbox[1, 1] + 1, bbox[0, 0]:bbox[0, 1] + 1, :] = sdf_var_uv_bbox_reshaped


    return sdf_values_uv, sdf_var_uv

def ray_integration_with_beta_dist(sdf_values, sdf_var, width, length, 
                                   valid_indices, num_depth_samples, th, sampled_depths):
    '''
    Given the mean and sigma of the sdf values of each sampled points, 
    calculate the depth image with beta distribution.

    :width: max y
    :length: max x

    :valid_indices: those points inside range 1.0 and have SDF values.

    '''

    # use valid indices, get a shape for sigma and mean as (ind_x, ind_y, n_samples_per_ray)
    # for those w/o any sdf values, we set it to be nan?

    # valid_indices are generated from (50176,50,...)
    # change 50176 back to (224,224)
    
    sdf_values_uv, sdf_var_uv = get_uv_ray_distribution(sdf_values, sdf_var,
                            width, length,
                            valid_indices, num_depth_samples)

    # now we get size (length, width, n_samples_per_ray); Consider each pixels to do the following calculation
    # go through each pixel, calculate the beta distribution

    # Debug, get a center line
    i = 112
    j = 112
    depth_distribution = compute_depth_distribution(sdf_values_uv[i, j, :], sdf_var_uv[i, j, :], th, sampled_depths)

# an uncertainty-aware version
def render_depth_image_with_uncertainty(decoder, latent_vector_distribution,
                                        K, t_obj_cam, width, length, num_depth_samples=50, keep_grad=True):
    # step 1: generate sampled points
    output = generate_sample_points(K, width, length, t_obj_cam, num_depth_samples)
    sampled_points_obj = output['sampled_points_obj']
    sampled_ray_depth = output['sampled_ray_depth']

    # select valid query points, and reshape it into (N, 3)
    # (num_rays, num_samples_per_ray)
    valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)
    # (n_valid, 3) flattened = (n_rays, n_depth_sample, 3)[x, y, :]
    query_points_obj = sampled_points_obj[valid_indices[0], valid_indices[1], :]


    # step 2: calculate sdfs, with uncertainty
    debug_mode = True  # if open, load from disk
    stored_sdfs_name = 'stored_sdfs.pt'
    stored_sdfs_exist = os.path.exists(stored_sdfs_name)
    if debug_mode and stored_sdfs_exist:
        stored_sdfs = torch.load(stored_sdfs_name)
        sdf_values, sdf_var = stored_sdfs['sdf_values'], stored_sdfs['sdf_var']
        print('[debug] load stored sdfs from disk')
    else:
        sdf_values, sdf_var = decode_uncertainty_sdf(decoder, latent_vector_distribution, query_points_obj, 
                                                 sample_num=10, keep_grad=keep_grad)
        if not stored_sdfs_exist:
            torch.save({'sdf_values': sdf_values, 'sdf_var': sdf_var}, stored_sdfs_name)

    # step 3: render depth with beta distribution
    # Now we begin using beta distributions!
    im_depth, im_var = ray_integration_with_beta_dist(sdf_values, sdf_var, width, length, 
                                                      valid_indices, num_depth_samples, th = 0.1, sampled_depths=sampled_ray_depth)

    print('debug here.')

# Render a depth image with given camera param, latent code
def render_depth_image(decoder, latent_vector, K, t_obj_cam, width, length, num_depth_samples=50):
    '''
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    num_depth_samples: for kitti config, 50
    '''
    
    '''
    Sample ray directions and ray depths
    '''
    # TODO: generate ray_directions, t_obj_cam, sampled_ray_depth
    xs = torch.arange(length)
    ys = torch.arange(width)
    uvs = torch.meshgrid(xs, ys) # generate a whole image pixels
    uvs = torch.stack(uvs).permute(1,2,0)
    uvs_homo = torch.cat([uvs, torch.ones(uvs.shape[0],uvs.shape[1], 1)], dim=-1)
    uvs_homo_flat = uvs_homo.flatten(0,1)
    K_inv = torch.Tensor(K).inverse()
    # (n, 3) = (n, 1, 3) * (3, 3)
    ray_directions = (uvs_homo_flat[:, None, :] * K_inv).sum(-1)  # an origin version. TODO: understand it
    # ray_directions = (uvs_homo_flat[:, None, :] @ K_inv).squeeze(1) # An intutive version
    
    # sampled_ray_depth
    t_cam_obj = torch.inverse(t_obj_cam)
    scale = torch.det(t_cam_obj[:3, :3]) ** (1 / 3)
    depth_min, depth_max = t_cam_obj[2, 3] - 1.0 * scale, t_cam_obj[2, 3] + 1.0 * scale
    sampled_ray_depth = torch.linspace(depth_min, depth_max, num_depth_samples).cuda()
    
    ray_directions = ray_directions.cuda()
    t_obj_cam = t_obj_cam.cuda()
    latent_vector = torch.Tensor(latent_vector).cuda()
    
    coordinates_uv = uvs_homo_flat[:,:2].int()
    output = compute_render_depth(decoder, ray_directions, t_obj_cam, sampled_ray_depth, \
        latent_vector, th=0.01, coordinates_uv=coordinates_uv) 

    if output is None:
        return None, None
    else:
        depth_values, depth_var, coordinates_uv_out = output

    # depth_values to images
    depth_values = depth_values.cpu()
    depth_var = depth_var.cpu()
    depth_im = np.full((width, length), 0, dtype=np.float16)

    coordinates_uv_out = coordinates_uv_out.cpu()
    
    # TODO: Check meaning of x,y
    # depth_im[coordinates_uv_out[:,0], coordinates_uv_out[:,1]] = depth_values
    depth_im[coordinates_uv_out[:,1], coordinates_uv_out[:,0]] = depth_values
    # var map
    var_im = np.full((width, length), 0, dtype=np.uint8)
    # var_im[coordinates_uv_out[:,0], coordinates_uv_out[:,1]] = depth_var
    var_im[coordinates_uv_out[:,1], coordinates_uv_out[:,0]] = depth_var
    
    return depth_im, var_im
