#
# This file is part of https://github.com/TRAILab/UncertainShapePose (The uncertainty part)
#  and also part of https://github.com/JingwenWang95/DSP-SLAM (The deterministic part)
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
from reconstruct.loss_utils import decode_sdf, get_batch_sdf_jacobian, get_points_to_pose_jacobian_se3, get_batch_sdf_values,\
    get_points_to_pose_jacobian_sim3, get_points_to_points_jacobian, get_points_to_scale_jacobian, sdf_to_occupancy
from reconstruct.utils import ForceKeyErrorDict

from uncertainty.energy_score import loss_energy_score_batch_mu_var

from uncertainty.probabilistic_rendering.loss_2d_uncertainty_utils_torch import depth_render, termination_probability_params_from_logitn

import numpy as np
import time

temp_output_list = []

def skew(w):
    wc = torch.stack((torch.tensor(0, dtype=torch.float32).cuda(), -w[2], w[1],
                      w[2], torch.tensor(0, dtype=torch.float32).cuda(), -w[0],
                      -w[1], w[0], torch.tensor(0, dtype=torch.float32).cuda()
                      )).view(3, 3)
    return wc


def Oplus(T, v):
    rho = v[:3] # translation
    phi = v[3:] # rotation
    tolerance = 1e-12

    # C = vec2rot(phi)
    ####################################################################
    angle = torch.norm(phi, p=2, dim=0)
    if angle < tolerance:
        # vec2rotSeries
        N = 10
        C = torch.eye(3, dtype=torch.float32).cuda()
        xM = torch.eye(3, dtype=torch.float32).cuda()
        cmPhi = skew(phi)
        for n in range(1, N+1):
            xM = torch.mm(xM, (cmPhi / n))
            C = C + xM
        tmp = sqrtm(torch.mm(torch.transpose(C,0,1), C))
        C = torch.mm(C, torch.inverse(tmp))
    else:
        axis_ = phi/angle
        axis = torch.reshape(axis_, (3, 1))
        cp = torch.cos(angle)
        sp = torch.sin(angle)
        I = torch.eye(3, dtype=torch.float32).cuda()
        C = cp * I + (1 - cp) * \
            torch.mm(axis,torch.transpose(axis,0,1)) + sp * skew(axis_)
    ####################################################################

    # J = vec2jac(phi)
    ####################################################################
    ph = torch.norm(phi, p=2, dim=0)
    if ph < tolerance:
        # vec2jacSeries
        N = 10
        J = torch.eye(3, dtype=torch.float32).cuda()
        pxn = torch.eye(3, dtype=torch.float32).cuda()
        px = skew(phi)
        for n in range(1, N+1):
            pxn = torch.mm(pxn, px) / (n+1)
            J = J + pxn
    else:
        axis_ = phi/ph
        axis = torch.reshape(axis_, (3, 1))
        cph = (1 - torch.cos(ph))/ph
        sph = torch.sin(ph)/ph
        I = torch.eye(3, dtype=torch.float32).cuda()
        J = sph * I + (1 - sph) * torch.mm(axis,torch.transpose(axis,0,1)) + cph * skew(axis_)

    rho_ = torch.reshape(rho, (3, 1))
    trans = torch.mm(J, rho_)
    dT = torch.stack((C[0, 0], C[0, 1], C[0, 2], trans[0,0],
                      C[1, 0], C[1, 1], C[1, 2], trans[1,0],
                      C[2, 0], C[2, 1], C[2, 2], trans[2,0],
                      torch.tensor(0, dtype=torch.float32).cuda(),
                      torch.tensor(0, dtype=torch.float32).cuda(),
                      torch.tensor(0, dtype=torch.float32).cuda(),
                      torch.tensor(1, dtype=torch.float32).cuda())).view(4, 4)

    return torch.mm(dT, T)


def sample_with_reparameterization(latent_vector_distribution, sample_num, logvar_to_sigma=True, device=None):
    if device is None:
        device = latent_vector_distribution.device
        
    dims = latent_vector_distribution.shape[0]
    normal_samples = torch.normal(torch.zeros(sample_num,dims), torch.ones(sample_num,dims)).to(device)
    mean_expand = latent_vector_distribution[:,0].unsqueeze(0).expand(sample_num,-1)
    sigma = latent_vector_distribution[:,1]
    if logvar_to_sigma:
        sigma = sigma.exp().sqrt()

    sigma_expand_mat = sigma.diag().unsqueeze(0).expand(sample_num,-1,-1)

    latent_codes_samples = (sigma_expand_mat @ normal_samples.unsqueeze(-1)).squeeze(-1) + mean_expand
    
    return latent_codes_samples


def compute_sdf_loss_shape_pose(decoder, data_surface_world, latent_vector_distribution, T_ow, T_ow_1, dPose_std, dScale_std,
                                loss_type="normal", logvar_to_sigma=True, clamp_dist=0.1,
                                points_sample_each_iter=1000, T_vis=None):
    '''
    @loss_type: if normal, we only use the origin dsp-slam method; if energy_score, we propagate uncertainty with jacobian

    @points_sample_each_iter: Update for Scannet; Sample N points for each iteration
    '''
    time_0 = time.time()

    # update: consider the case where pts_surface_cam contain gt sdf values
    if len(data_surface_world) < points_sample_each_iter:
        pts_surface_world = data_surface_world  # get all
    else:
        pts_surface_world = data_surface_world[np.random.choice(len(data_surface_world), points_sample_each_iter, replace=False)]
    sdf_values_ob = torch.zeros((pts_surface_world.shape[0], 1))

    # robust filtering: ignore nan values
    non_nan_indices = ~sdf_values_ob.isnan().squeeze()
    sdf_values_ob = sdf_values_ob[non_nan_indices, :]
    pts_surface_world = pts_surface_world[non_nan_indices, :]

    # transform points to object coordinates; The two are the same, for gradients calculation
    ###############################################################################################
    pts_surface_obj = torch.mm(T_ow, pts_surface_world.permute(1, 0)).permute(1, 0)[:, :3]
    # For Jacobian Use
    x_obj = (pts_surface_world[..., None, :3] * T_ow.cuda()[:3, :3]).sum(-1) + T_ow.cuda()[:3, 3]
    ###############################################################################################

    # Update: Clamp the points areas to be within 1.0
    pts_surface_obj = torch.clamp(pts_surface_obj, -1.0, 1.0)
    x_obj = torch.clamp(x_obj, -1.0, 1.0)

    # Go through decoder functions
    ###################################################################
    device = latent_vector_distribution.device
    # clip sdf values
    sdf_values_ob = torch.clamp(sdf_values_ob, -clamp_dist, clamp_dist)   # useless; since all are 0.
    # if loss_type is normal, we do not sample and only use the mean!
    latent_codes_samples = latent_vector_distribution[:,0].unsqueeze(0)
    # (sample_num, points_num, 1)
    n_points = pts_surface_obj.shape[0]
    latent_repeat = latent_codes_samples.expand(n_points, -1)
    inputs = torch.cat([latent_repeat, pts_surface_obj], -1)
    sdf_mean = decoder(inputs)
    # Attention: when there is uncertainty, how to clamp theoretically?
    #sdf_mean = torch.clamp(sdf_mean, -clamp_dist, clamp_dist)
    ####################################################################
    time_1 = time.time()

    b_use_uncertainty = loss_type in ['energy_score', 'nll']
    if b_use_uncertainty: # propagate uncertainty only when using it to save time
        # Jacobian related
        #############################################################################
        out_dim_ = 1
        n_ = x_obj.shape[0]
        input_x_ = x_obj.clone().detach()
        lat_vec_ = latent_vector_distribution[:,0].clone().detach()
        latent_repeat_ = lat_vec_.expand(n_, -1)
        inputs_ = torch.cat([latent_repeat_, input_x_], 1)
        inputs_ = inputs_.unsqueeze(1)  # (n, 1, in_dim)
        inputs_ = inputs_.repeat(1, out_dim_, 1)
        inputs_.requires_grad = True
        y_ = decoder(inputs_)  # (n, out_dim, out_dim)
        #y_ = torch.clamp(y_, -clamp_dist, clamp_dist)
        w_ = torch.eye(out_dim_).view(1, out_dim_, out_dim_).repeat(n_, 1, 1).cuda()
        y_.backward(w_, retain_graph=False)
        de_di = inputs_.grad.data.detach()
        # Jacobian for SDF to code
        jac_code = de_di[..., :-3]
        # Jacobian for SDF to object point
        de_dxo = de_di[..., -3:]
        # Jacobian for object point to 6D pose
        dxo_dtow = get_points_to_pose_jacobian_se3(x_obj)  # xo = toc @ xc      # CHECK: x_c?
        # Jacobian for object point to scale point
        dxo_dxs = get_points_to_points_jacobian(x_obj, T_ow_1)  # xo = tow (no scale) @ xs
        # Jacobian for scale point to 3D scale
        dxs_ds = get_points_to_scale_jacobian(pts_surface_world)    # xs = s @ xw
        # Jacobian for SDF to 6D pose
        jac_tow = torch.bmm(de_dxo, dxo_dtow)
        # Jacobian for SDF to 3D scale
        jac_scale = torch.bmm(de_dxo, torch.bmm(dxo_dxs, dxs_ds))
        #############################################################################
        # Full Jacobian
        jac_full = torch.cat([jac_code, jac_tow, jac_scale], -1)
        time_2 = time.time()

        # Compute SDF uncertainty and loss
        ####################################################################################
        ## Code uncertainty only
        #############################################################################
        # code_dis = torch.exp(latent_vector_distribution[:, 1])
        # code_dis = torch.diag(code_dis)
        # code_dis = code_dis.view(1, 64, 64).repeat(n_, 1, 1)
        # sdf_unc = torch.bmm(jac_code, torch.bmm(code_dis, jac_code.permute(0, 2, 1)))
        #############################################################################
        code_pose_dis = torch.cat([torch.exp(latent_vector_distribution[:, 1]),
                                torch.exp(dPose_std),
                                torch.exp(dScale_std)], -1)
        code_pose_dis = torch.diag(code_pose_dis)  # covariance matrix
        code_pose_dis = code_pose_dis.view(1, 73, 73).repeat(n_, 1, 1)
        sdf_unc = torch.bmm(jac_full, torch.bmm(code_pose_dis, jac_full.permute(0, 2, 1)))
        ####################################################################################
        time_3 = time.time()

        # sdf loss: NLL loss for all points
        sdf_var = sdf_unc.squeeze(1)

    sdf_values_ob = sdf_values_ob.to(device)
    goals = sdf_values_ob
       
    normal_loss = ((sdf_mean - sdf_values_ob).square()).mean(0)
    if loss_type == 'energy_score':
        sdf_loss = loss_energy_score_batch_mu_var(sdf_mean, sdf_var, goals, M_sample=1000)
    elif loss_type == 'normal':
        sdf_loss = normal_loss
    elif loss_type == 'nll':
        # elif loss_type == 'NLL_pure':
        #     loss = ((means - goals).square() / variances).mean()
        sdf_loss = ((sdf_mean - sdf_values_ob).square() / sdf_var + sdf_var.log()).mean(0)
    pts_output = (pts_surface_world @ T_vis.T)[:, :3].clone()  # points in camera views

    time_4 = time.time()

    return ForceKeyErrorDict(loss=sdf_loss, pts_local=pts_output, normal_loss=normal_loss)


# if latent_vector_distribution stores uncertainty as log var, open logvar_to_sigma
def compute_sdf_loss_uncertain(decoder, data_surface_cam, t_obj_cam, 
                               latent_vector_distribution, sample_num = 10, loss_type = "normal", logvar_to_sigma=True, 
    clamp_dist = 0.1):
    """
    :param decoder: DeepSDF decoder
    :param data_surface_cam: surface points under camera coordinate (N, 3)
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param latent_vector_distribution: shape code distribution (dims, 2); Attention: uncertainty is log(var)
    :return: Jacobian wrt pose (N, 1, 7), Jacobian wrt shape code (N, 1, code_len), error residuals (N, 1, 1)
    """
    
    if data_surface_cam.shape[1] == 4:
        pts_surface_cam = data_surface_cam[:,:3]
        sdf_values_ob = data_surface_cam[:,-1:]
    else:
        pts_surface_cam = data_surface_cam
        sdf_values_ob = torch.zeros((pts_surface_cam.shape[0],1))

    # robust filtering: ignore nan values
    non_nan_indices = ~sdf_values_ob.isnan().squeeze()
    sdf_values_ob = sdf_values_ob[non_nan_indices, :]
    pts_surface_cam = pts_surface_cam[non_nan_indices, :]

    # transform points to object coordinates
    pts_surface_obj = \
        (pts_surface_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]

    # sample multiple times from code distribution, then go through those functions
    device = latent_vector_distribution.device

    # clip sdf values 
    sdf_values_ob = torch.clamp(sdf_values_ob, -clamp_dist, clamp_dist)

    # if loss_type is normal, we do not sample and only use the mean!
    if loss_type == 'normal' or loss_type == 'l1':
        latent_codes_samples = latent_vector_distribution[:,0].unsqueeze(0)
    else:
        # when using uncertainty
        use_reparameterization_tricks = True
        if use_reparameterization_tricks:
            latent_codes_samples = sample_with_reparameterization(latent_vector_distribution, sample_num, logvar_to_sigma, device)

        else:
            # Checked and find the gradients are 0 if directly use it.
            mean_expand = latent_vector_distribution[:,0].unsqueeze(0).expand(sample_num,-1)
            sigma = latent_vector_distribution[:,1]
            if logvar_to_sigma:
                sigma = sigma.exp().sqrt()

            sigma_expand = sigma.unsqueeze(0).expand(sample_num,-1)
            latent_codes_samples = torch.normal(mean_expand, sigma_expand)

    # (sample_num, points_num, 1)
    sdf_batch = get_batch_sdf_values(decoder, latent_codes_samples, pts_surface_obj)

    # calculate sample mean and sample var
    # (points_num, 1)
    sdf_mean = sdf_batch.mean(0)
    sdf_var = sdf_batch.var(0)

    sdf_mean = torch.clamp(sdf_mean, -clamp_dist, clamp_dist)

    sdf_values_ob = sdf_values_ob.to(device)
    normal_loss = ( (sdf_mean - sdf_values_ob).square() ).mean(0)
    if loss_type == "NLL":
        sdf_loss = ((sdf_mean - sdf_values_ob).square() / sdf_var + sdf_var.log()).mean(0)
    elif loss_type == "normal":
        sdf_loss = normal_loss
    elif loss_type == "energy_score":
        goals = sdf_values_ob
        sdf_loss = loss_energy_score_batch_mu_var(sdf_mean, sdf_var, goals, M_sample=1000)
    elif loss_type == 'l1':
        loss_l1 = torch.nn.L1Loss()
        sdf_loss = loss_l1(sdf_mean, sdf_values_ob)

    return ForceKeyErrorDict(loss=sdf_loss, pts_local=pts_surface_obj, normal_loss=normal_loss)

def compute_sdf_loss(decoder, pts_surface_cam, t_obj_cam, latent_vector):
    """
    :param decoder: DeepSDF decoder
    :param pts_surface_cam: surface points under camera coordinate (N, 3)
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param latent_vector: shape code
    :return: Jacobian wrt pose (N, 1, 7), Jacobian wrt shape code (N, 1, code_len), error residuals (N, 1, 1)
    """
    # (n_sample_surface, 3)
    pts_surface_obj = \
        (pts_surface_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]

    res_sdf, de_di = get_batch_sdf_jacobian(decoder, latent_vector, pts_surface_obj, 1)
    # SDF term Jacobian
    de_dxo = de_di[..., -3:]  ## Only for x,y,z
    # Jacobian for pose
    dxo_dtoc = get_points_to_pose_jacobian_sim3(pts_surface_obj)
    jac_toc = torch.bmm(de_dxo, dxo_dtoc)
    # Jacobian for code
    jac_code = de_di[..., :-3]

    return jac_toc, jac_code, res_sdf


def compute_render_loss(decoder, ray_directions, depth_obs, t_obj_cam, sampled_ray_depth, latent_vector, th=0.01):
    """
    :param decoder: DeepSDF decoder
    :param ray_directions: (N, 3) under camera coordinate
    :param depth_obs: (N,) observed depth values for foreground pixels, 1.1 * d_max for background pixels
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param sampled_ray_depth: (M,) linspace between d_min and d_max
    :param latent_vector: shape code
    :param th: cut-off threshold for converting SDF to occupancy
    :return: Jacobian wrt pose (K, 1, 7), Jacobian wrt shape code (K, 1, code_len), error residuals (K, 1, 1)
    Note that K is the number of points that have non-zero Jacobians out of a total of N * M points
    """

    # (n_rays, num_samples_per_ray, 3) = (n_rays, 1, 3) * (num_samples_per_ray, 1)
    sampled_points_cam = ray_directions[..., None, :] * sampled_ray_depth[:, None]
    # (n_rays, num_samples_per_ray, 3)
    sampled_points_obj = \
        (sampled_points_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]
    n_rays = sampled_points_obj.shape[0]
    n_depths = sampled_ray_depth.shape[0]

    # (num_rays, num_samples_per_ray)
    valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)
    # (n_valid, 3) flattened = (n_rays, n_depth_sample, 3)[x, y, :]
    query_points_obj = sampled_points_obj[valid_indices[0], valid_indices[1], :]

    # if too few query points, return immediately
    if query_points_obj.shape[0] < 10:
        return None

    # flattened
    with torch.no_grad():
        sdf_values = decode_sdf(decoder, latent_vector, query_points_obj).squeeze()

    if sdf_values is None:
        raise Exception("no valid query points?")

    # Full dimension (n_rays, n_samples_per_ray)
    occ_values = torch.full((n_rays, n_depths), 0.).cuda()
    valid_indices_x, valid_indices_y = valid_indices  # (indices on x, y dimension)
    occ_values[valid_indices_x, valid_indices_y] = sdf_to_occupancy(sdf_values, th=th)

    with_grad = (sdf_values > -th) & (sdf_values < th)
    with_grad_indices_x = valid_indices_x[with_grad]
    with_grad_indices_y = valid_indices_y[with_grad]

    # point-wise values, i.e. multiple points might belong to one pixel (m, n_samples_per_ray)
    occ_values_with_grad = occ_values[with_grad_indices_x, :]
    m = occ_values_with_grad.shape[0]  # number of points with grad
    d_min = sampled_ray_depth[0]
    d_max = sampled_ray_depth[-1]

    # Render function
    acc_trans = torch.cumprod(1 - occ_values_with_grad, dim=-1)
    acc_trans_augment = torch.cat(
        (torch.ones(m, 1).cuda(), acc_trans),
        dim=-1
    )
    o = torch.cat(
        (occ_values_with_grad, torch.ones(m, 1).cuda()),
        dim=-1
    )
    d = torch.cat(
        (sampled_ray_depth, torch.tensor([1.1 * d_max]).cuda()),
        dim=-1
    )
    term_prob = (o * acc_trans_augment)
    # rendered depth values (m,)
    d_u = torch.sum(d * term_prob, dim=-1)
    var_u = torch.sum(term_prob * (d[None, :] - d_u[:, None]) ** 2, dim=-1)

    # Get Jacobian of depth residual wrt occupancy probability de_do
    o_k = occ_values[with_grad_indices_x, with_grad_indices_y]
    l = torch.arange(n_depths).cuda()
    l = l[None, :].repeat(m, 1)
    acc_trans[l < with_grad_indices_y[:, None]] = 0.
    de_do = acc_trans.sum(dim=-1) / (1. - o_k)

    # Remove points with zero gradients, and get de_ds = de_do * do_ds
    non_zero_grad = (de_do > 1e-2)
    de_do = de_do[non_zero_grad]
    d_u = d_u[non_zero_grad]
    delta_d = (d_max - d_min) / (n_depths - 1)
    do_ds = -1. / (2 * th)
    de_ds = (de_do * delta_d * do_ds).view(-1, 1, 1)

    # get residuals
    with_grad_indices_x = with_grad_indices_x[non_zero_grad]
    with_grad_indices_y = with_grad_indices_y[non_zero_grad]
    depth_obs_non_zero_grad = depth_obs[with_grad_indices_x]  # (m,)
    res_d = depth_obs_non_zero_grad - d_u  # (m,)

    # make it more robust and stable
    res_d[res_d > 0.30] = 0.30
    res_d[res_d < -0.30] = -0.30
    res_d = res_d.view(-1, 1, 1)

    pts_with_grad = sampled_points_obj[with_grad_indices_x, with_grad_indices_y]
    _, ds_di = get_batch_sdf_jacobian(decoder, latent_vector, pts_with_grad, 1)
    de_di = de_ds * ds_di  # (m, 1, code_len + 3)
    de_dxo = de_di[..., -3:]  # (m, 1, 3)
    # Jacobian for pose and code
    dxo_dtoc = get_points_to_pose_jacobian_sim3(pts_with_grad)
    jac_toc = torch.bmm(de_dxo, dxo_dtoc)
    jac_code = de_di[..., :-3]  # (m, 1, code_len)

    return jac_toc, jac_code, res_d

def compute_render_depth(decoder, ray_directions, t_obj_cam, sampled_ray_depth, latent_vector, th=0.01,
                        coordinates_uv=None):
    """
    Return rendered depth only.
    
    :param decoder: DeepSDF decoder
    :param ray_directions: (N, 3) under camera coordinate
    :param depth_obs: (N,) observed depth values for foreground pixels, 1.1 * d_max for background pixels
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param sampled_ray_depth: (M,) linspace between d_min and d_max
    :param latent_vector: shape code
    :param th: cut-off threshold for converting SDF to occupancy
    :param coordinates_uv: record the coordinates with filtering
    :return: Jacobian wrt pose (K, 1, 7), Jacobian wrt shape code (K, 1, code_len), error residuals (K, 1, 1)
    Note that K is the number of points that have non-zero Jacobians out of a total of N * M points
    """
    
    # TODO: We sample multiple times and add them together!
    

    # (n_rays, num_samples_per_ray, 3) = (n_rays, 1, 3) * (num_samples_per_ray, 1)
    sampled_points_cam = ray_directions[..., None, :] * sampled_ray_depth[:, None]
    # (n_rays, num_samples_per_ray, 3)
    sampled_points_obj = \
        (sampled_points_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]
    n_rays = sampled_points_obj.shape[0]
    n_depths = sampled_ray_depth.shape[0]

    ## Check that under object coordinate, the points should inside a normalized area 1. If not, the transformation is not correct
    # (num_rays, num_samples_per_ray)
    valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)
    # (n_valid, 3) flattened = (n_rays, n_depth_sample, 3)[x, y, :]
    query_points_obj = sampled_points_obj[valid_indices[0], valid_indices[1], :]
    
    # if too few query points, return immediately
    if query_points_obj.shape[0] < 10:
        return None

    # flattened
    with torch.no_grad():
        sdf_values = decode_sdf(decoder, latent_vector, query_points_obj).squeeze()

    if sdf_values is None:
        raise Exception("no valid query points?")

    # Full dimension (n_rays, n_samples_per_ray)
    occ_values = torch.full((n_rays, n_depths), 0.).cuda()
    valid_indices_x, valid_indices_y = valid_indices  # (indices on x, y dimension)
    occ_values[valid_indices_x, valid_indices_y] = sdf_to_occupancy(sdf_values, th=th)
    # coordinates_uv = coordinates_uv[]

    with_grad = (sdf_values > -th) & (sdf_values < th)
    with_grad_indices_x = valid_indices_x[with_grad]
    with_grad_indices_y = valid_indices_y[with_grad]

    # point-wise values, i.e. multiple points might belong to one pixel (m, n_samples_per_ray)
    occ_values_with_grad = occ_values[with_grad_indices_x, :]
    m = occ_values_with_grad.shape[0]  # number of points with grad
    d_min = sampled_ray_depth[0]
    d_max = sampled_ray_depth[-1]

    # Render function
    acc_trans = torch.cumprod(1 - occ_values_with_grad, dim=-1)
    acc_trans_augment = torch.cat(
        (torch.ones(m, 1).cuda(), acc_trans),
        dim=-1
    )
    o = torch.cat(
        (occ_values_with_grad, torch.ones(m, 1).cuda()),
        dim=-1
    )
    d = torch.cat(
        (sampled_ray_depth, torch.tensor([1.1 * d_max]).cuda()),
        dim=-1
    )
    term_prob = (o * acc_trans_augment)
    # rendered depth values (m,)
    d_u = torch.sum(d * term_prob, dim=-1)
    var_u = torch.sum(term_prob * (d[None, :] - d_u[:, None]) ** 2, dim=-1)
    
    if coordinates_uv is not None:
        coordinates_uv = coordinates_uv.to(with_grad_indices_x.device)
        coordinates_uv = coordinates_uv[with_grad_indices_x, ...]    
        return d_u, var_u, coordinates_uv
    else:
        return d_u, var_u

def compute_render_loss_nojac(decoder, ray_directions, depth_obs, t_obj_cam, sampled_ray_depth, latent_vector, th=0.01):
    """
    A version w/o Jacobian calculation to save computation.
    
    :param decoder: DeepSDF decoder
    :param ray_directions: (N, 3) under camera coordinate
    :param depth_obs: (N,) observed depth values for foreground pixels, 1.1 * d_max for background pixels
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param sampled_ray_depth: (M,) linspace between d_min and d_max
    :param latent_vector: shape code
    :param th: cut-off threshold for converting SDF to occupancy
    :return: Jacobian wrt pose (K, 1, 7), Jacobian wrt shape code (K, 1, code_len), error residuals (K, 1, 1)
    Note that K is the number of points that have non-zero Jacobians out of a total of N * M points
    """
    
    # (n_rays, num_samples_per_ray, 3) = (n_rays, 1, 3) * (num_samples_per_ray, 1)
    sampled_points_cam = ray_directions[..., None, :] * sampled_ray_depth[:, None]
    # (n_rays, num_samples_per_ray, 3)
    sampled_points_obj = \
        (sampled_points_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]
    n_rays = sampled_points_obj.shape[0]
    n_depths = sampled_ray_depth.shape[0]

    # (num_rays, num_samples_per_ray)
    # For those sampled points in object frame, only consider those inside 1.0 range;
    # since only those points are trained for DeepSDF
    valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)
    
    # (n_valid, 3) flattened = (n_rays, n_depth_sample, 3)[x, y, :]
    # select valid points from all sampled points
    query_points_obj = sampled_points_obj[valid_indices[0], valid_indices[1], :]

    # if too few query points, return immediately
    if query_points_obj.shape[0] < 10:
        raise ValueError

    sdf_values = decode_sdf(decoder, latent_vector, query_points_obj, keep_grad=True).squeeze()

    if sdf_values is None:
        raise Exception("no valid query points?")

    # Full dimension (n_rays, n_samples_per_ray)
    # initialize the full occupancy values for every sampled depth points on each ray;
    # note that some points on the ray do not have valid sdf values, we SET THEM AS 0.
    occ_values = torch.full((n_rays, n_depths), 0.).cuda()
    valid_indices_x, valid_indices_y = valid_indices  # (indices on x, y dimension)
    occ_values[valid_indices_x, valid_indices_y] = sdf_to_occupancy(sdf_values, th=th)

    # for those valid points inside range 1.0, we get SDF values, but some of them have larger than threshold;
    # those points are too far away or inside the surface, and we ignore them for gradient calculation;
    # Also ignore them for depth calculation
    with_grad = (sdf_values > -th) & (sdf_values < th)
    with_grad_indices_x = valid_indices_x[with_grad]
    with_grad_indices_y = valid_indices_y[with_grad]

    occ_values_with_grad = occ_values  # keep all the rays


    m = occ_values_with_grad.shape[0]  # number of points (pixels) with grad
    d_min = sampled_ray_depth[0]
    d_max = sampled_ray_depth[-1]

    # Render function
    # occ_values_with_grad_minus = (1 - occ_values_with_grad).clone()
    acc_trans = torch.cumprod(1 - occ_values_with_grad, dim=-1)
    acc_trans_augment = torch.cat(  # add 1 to the beginning
        (torch.ones(m, 1).cuda(), acc_trans),
        dim=-1
    )
    o = torch.cat(  # add 1 to the end
        (occ_values_with_grad, torch.ones(m, 1).cuda()),
        dim=-1
    )
    d = torch.cat(  # add d_max to the end
        (sampled_ray_depth, torch.tensor([1.1 * d_max]).cuda()),
        dim=-1
    )
    # calculate expectation
    term_prob = (o * acc_trans_augment)
    # rendered depth values (m,)
    d_u = torch.sum(d * term_prob, dim=-1)
    var_u = torch.sum(term_prob * (d[None, :] - d_u[:, None]) ** 2, dim=-1)  # TODO: Consider how to use this var

    depth_obs_non_zero_grad = depth_obs
    res_d = depth_obs_non_zero_grad - d_u  # (m,)

    # make it more robust and stable
    res_d[res_d > 0.30] = 0.30
    res_d[res_d < -0.30] = -0.30
    res_d_out = res_d.view(-1, 1, 1)

    var_u_out = var_u.view(-1, 1, 1)

    return res_d_out, var_u_out

def compute_render_loss_nojac_multiview(decoder, ray_directions_mv, 
                                        depth_obs_mv, t_obj_cam_mv, sampled_ray_depth_mv, latent_vector, th=0.01,
                                        loss_type_2d_uncertain='dsp'):
    '''
    an IO function for multi-view 2d rendering.
    '''

    # iterating every views, then stack together
    n_views = len(ray_directions_mv)

    res_d_list = []

    var_u_list = []
    for i in range(n_views):
        ray_directions = ray_directions_mv[i]
        depth_obs = depth_obs_mv[i]
        t_obj_cam = t_obj_cam_mv[i]

        sampled_ray_depth = sampled_ray_depth_mv[i]

        try:
            res_d, var_u = compute_render_loss_nojac(decoder, ray_directions, depth_obs, t_obj_cam, 
                                            sampled_ray_depth, latent_vector, th=th)
            res_d_list.append(res_d)
            var_u_list.append(var_u)
        except:
            print('Error: 2d loss')
            pass
    
    res_d_out = torch.cat(res_d_list, dim=0)
    var_u_out = torch.cat(var_u_list, dim=0)

    if loss_type_2d_uncertain == 'dsp':
        render_loss = torch.mean(res_d_out ** 2)
    elif loss_type_2d_uncertain == 'node' or loss_type_2d_uncertain == 'node_es':
        # only consider points with var > 0
        mask = var_u_out > 0
        res_d_out = res_d_out[mask]
        var_u_out = var_u_out[mask]

        # check if there is no points left
        if res_d_out.shape[0] == 0:
            # failed
            print('Warning: no valid 2d points with var > 0.')
            render_loss = torch.tensor(0.0, device=res_d_out.device, dtype=res_d_out.dtype)

        # final_loss_type = 'energy_score'  # nll
        final_loss_type = 'nll'
        if loss_type_2d_uncertain == 'node_es':
            final_loss_type = 'energy_score'  # nll

        if final_loss_type == 'nll':
            render_loss = (res_d_out.square() / var_u_out + var_u_out.log()).mean()
        elif final_loss_type == 'energy_score':
            goal = torch.zeros_like(res_d_out)
            render_loss = loss_energy_score_batch_mu_var(res_d_out.unsqueeze(-1), var_u_out.unsqueeze(-1), goal.unsqueeze(-1), M_sample=1000)

    return render_loss


def uncertain_render_beta_io(mus, sigmas, depths, 
                             device=torch.device("cuda"), dtype=torch.float64,
                             BACKGROUND_DEPTH = np.array([9.0]), k = 400,
                            const_a = None):
    '''

    An IO function to connect the scripts.

    :param mus: (pixel_u, pixel_v, sample_num) mean of the sdf values
    :param sigmas: (pixel_u, pixel_v, sample_num) std of the sdf values
    :param depths: (sample_num) depth values

    :return: (pixel_u, pixel_v) rendered depth values

    '''    
    ###
    # Method config
    max_ray_sample = 20

    # min_ray_sample = 2
    sdf_threshold = 0.025
    ###

    tensor_mus = mus
    tensor_sigmas = sigmas
    tensor_depths = depths

    Nu = mus.shape[0]
    Nv = mus.shape[1]
    N_coords = Nu * Nv

    output_depth = torch.zeros((mus.shape[0], mus.shape[1]), device=device, dtype=dtype)
    output_var = torch.zeros((mus.shape[0], mus.shape[1]), device=device, dtype=dtype)

    pixel_coords = torch.tensor([(u, v) for u in range(Nu) for v in range(Nv)], device=device, dtype=torch.int)

    BG_val = torch.tile(BACKGROUND_DEPTH.reshape((-1,1)), [1, max_ray_sample])

    batch_size = 4096
    for i in range(0, N_coords, batch_size):
        # print("Working on batch", int(i / batch_size)+1,"out of", int(np.ceil(N_coords / batch_size)))

        time_preprocess_start = time.time()

        batch_coords = pixel_coords[i:i+batch_size]

        us, vs = batch_coords[:, 0].long(), batch_coords[:, 1].long()

        batch_mus = tensor_mus[us, vs]
        batch_sigmas = tensor_sigmas[us, vs]
        batch_depths = tensor_depths.repeat(len(batch_coords), 1)

        mask_sdf = torch.abs(batch_mus) <= sdf_threshold
        
        # mask_pos = batch_mus > 0
        # mask_neg = batch_mus < 0
        shifted_mask_full, shift_indices_full = (mask_sdf.int() * 1).sort(dim=1, descending=True, stable=True)

        max_ray_sample_batch = max_ray_sample

        shifted_mask = shifted_mask_full[:,:max_ray_sample_batch].bool()
        shift_indices = shift_indices_full[:,:max_ray_sample_batch]

        shifted_mus = (batch_mus.gather(1, shift_indices) * shifted_mask)
        shifted_sigmas = (batch_sigmas.gather(1, shift_indices) * shifted_mask)
        shifted_depths = (batch_depths.gather(1, shift_indices) * shifted_mask)

        # shifted_mus[~shifted_mask] = BACKGROUND_DEPTH
        shifted_mus[~shifted_mask] = -sdf_threshold
        shifted_sigmas[~shifted_mask] = sdf_threshold / 10
        # shifted_sigmas[~shifted_mask] = sdf_threshold  # increase so that the background sigma is not too small
        
        # consider batch size, cut the shape of BG_val, and BACKGROUND_DEPTH
        batch_num_this = len(batch_coords)
        BG_val_cur = BG_val[:batch_num_this,:]
        shifted_depths[~shifted_mask] = BG_val_cur[~shifted_mask]

        BACKGROUND_DEPTH_cur = BACKGROUND_DEPTH[:,:batch_num_this]
        shifted_depths = torch.hstack((shifted_depths, BACKGROUND_DEPTH_cur.reshape((-1,1))))

        time_preprocess_end = time.time()
        # print("    Preprocess time:", time_preprocess_end - time_preprocess_start)

        time_render_start = time.time()

        term_alphas, term_betas = termination_probability_params_from_logitn(k, shifted_mus, shifted_sigmas)

        '''
        Add a mask to filter those larger than background depth
        '''
        b_open_mask_filter = False
        if b_open_mask_filter:

            mask = (shifted_depths >= BACKGROUND_DEPTH-0.01)
            term_alphas[mask] = 0
            term_betas[mask] = 0

        E_d, Var_d = depth_render(shifted_depths, term_alphas, term_betas)
        
        '''
        Calibrate
        '''
        if const_a is None or const_a < 1e-6:
            Var_d_calib = Var_d
        else:
            mean_ray_sigmas = torch.mean(shifted_sigmas, dim=1)
            Var_d_calib = Var_d * (mean_ray_sigmas ** 2) * (const_a**2)
    
        time_render_end = time.time()

        # print("    Render time:", time_render_end - time_render_start)

        time_postprocess_start = time.time()

        output_depth[us,vs] = E_d
        output_var[us,vs] = Var_d_calib

        time_postprocess_end = time.time()

        # print("    Postprocess time:", time_postprocess_end - time_postprocess_start)

    return output_depth, output_var


def render_uncertain_depth(mask, K, t_obj_cam, decoder, latent_vector_distribution, 
                           num_depth_samples=50, BACKGROUND_DEPTH=9.0,
                           device='cuda', dtype=torch.float64):
    # first, sample points in the mask.
    from reconstruct.render_loss import generate_sample_points, decode_uncertainty_sdf, get_uv_ray_distribution

    width, length = mask.shape

    mask=None

    t_obj_cam = torch.from_numpy(t_obj_cam).cuda().float()
    K = torch.from_numpy(K).cuda().float()
    output = generate_sample_points(K, width, length, t_obj_cam, num_depth_samples, uv=None, mask=mask)
    sampled_points_obj = output['sampled_points_obj']
    sampled_ray_depth = output['sampled_ray_depth']

    valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)
    # (n_valid, 3) flattened = (n_rays, n_depth_sample, 3)[x, y, :]
    query_points_obj = sampled_points_obj[valid_indices[0], valid_indices[1], :].cuda()

    keep_grad = False
    # if latent_vector_distribution is numpy, change to torch and to device
    if isinstance(latent_vector_distribution, np.ndarray):
        latent_vector_distribution = torch.from_numpy(latent_vector_distribution).cuda().float()
    sdf_values, sdf_var = decode_uncertainty_sdf(decoder, latent_vector_distribution, query_points_obj, 
                                                sample_num=10, keep_grad=keep_grad)

    # choose 
    sdf_values_uv, sdf_var_uv = get_uv_ray_distribution(sdf_values, sdf_var,
                            width, length,
                            valid_indices, num_depth_samples, mask=mask)


    sdf_mean_im = sdf_values_uv
    sdf_std_im = torch.sqrt(sdf_var_uv)


    # if all dim 2 are == 1.0, we ignore it
    valid_mask_image = ~ ((sdf_values_uv == 1.0).all(-1))

    sdf_mean_im_valid = sdf_mean_im[valid_mask_image].unsqueeze(1).to(device)
    sdf_std_im_valid = sdf_std_im[valid_mask_image].unsqueeze(1).to(device)

    # (pixel_u, pixel_v)
    E_d_valid, Var_d_valid = uncertain_render_beta_io(sdf_mean_im_valid, sdf_std_im_valid, sampled_ray_depth, 
                                          device=device, dtype=dtype,
                                          BACKGROUND_DEPTH=BACKGROUND_DEPTH)

    # recover a full image
    E_d = torch.full((length, width), BACKGROUND_DEPTH, dtype=dtype)
    Var_d = torch.full((length, width), 0.0, dtype=dtype)
    E_d[valid_mask_image] = E_d_valid.detach().cpu().squeeze()
    Var_d[valid_mask_image] = Var_d_valid.detach().cpu().squeeze()
    
    # transpose
    E_d = E_d.T
    Var_d = Var_d.T

    return E_d, Var_d

def compute_render_loss_uncertain_beta(decoder, ray_directions, depth_obs, 
                                       t_obj_cam, sampled_ray_depth, latent_vector_distribution, 
                                       th=0.01, sample_num=10,
                                       loss_type='energy_score',
                                       dtype=torch.float64,
                                       BACKGROUND_DEPTH=9.0):
    """
    Method: Use beta distribution to model rendering process.

    :param decoder: DeepSDF decoder
    :param ray_directions: (N, 3) under camera coordinate
    :param depth_obs: (N,) observed depth values for foreground pixels, 1.1 * d_max for background pixels
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param sampled_ray_depth: (M,) linspace between d_min and d_max
    :param latent_vector: shape code
    :param th: cut-off threshold for converting SDF to occupancy
    :return: Jacobian wrt pose (K, 1, 7), Jacobian wrt shape code (K, 1, code_len), error residuals (K, 1, 1)
    Note that K is the number of points that have non-zero Jacobians out of a total of N * M points
    """
    # (n_rays, num_samples_per_ray, 3) = (n_rays, 1, 3) * (num_samples_per_ray, 1)
    sampled_points_cam = ray_directions[..., None, :] * sampled_ray_depth[:, None]
    # (n_rays, num_samples_per_ray, 3)
    sampled_points_obj = \
        (sampled_points_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]
    n_rays = sampled_points_obj.shape[0]
    n_depths = sampled_ray_depth.shape[0]

    # (num_rays, num_samples_per_ray)
    # For those sampled points in object frame, only consider those inside 1.0 range;
    # since only those points are trained for DeepSDF
    valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)
    
    # (n_valid, 3) flattened = (n_rays, n_depth_sample, 3)[x, y, :]
    # select valid points from all sampled points
    query_points_obj = sampled_points_obj[valid_indices[0], valid_indices[1], :]

    # if too few query points, return immediately
    if query_points_obj.shape[0] < 10:
        return None

    # sdf_values = decode_sdf(decoder, latent_vector, query_points_obj, keep_grad=True).squeeze()
    logvar_to_sigma = True
    device = latent_vector_distribution.device
    latent_codes_samples = sample_with_reparameterization(latent_vector_distribution, 
                                                          sample_num, logvar_to_sigma, device)
    # (sample_num, points_num, 1)
    pts_surface_obj = query_points_obj   # (sample_num, code_dim)
    sdf_batch = get_batch_sdf_values(decoder, latent_codes_samples, pts_surface_obj)

    # calculate sample mean and sample var
    # (points_num, )
    sdf_mean = sdf_batch.mean(0).squeeze()
    sdf_std = sdf_batch.std(0).squeeze()
    print(sdf_std.min(), sdf_std.max())

    # Full dimension (n_rays, n_samples_per_ray)
    SDF_INVALID = 1.0 # for those outside of the range (CONFIRMED the same as before)
    SIGMA_INVALID = 1.0

    sdf_mean_full = torch.full((n_rays, n_depths), SDF_INVALID).cuda()
    sdf_std_full = torch.full((n_rays, n_depths), SIGMA_INVALID).cuda()
    valid_indices_x, valid_indices_y = valid_indices  # (indices on x, y dimension)
    sdf_mean_full[valid_indices_x, valid_indices_y] = sdf_mean
    sdf_std_full[valid_indices_x, valid_indices_y] = sdf_std

    # render with beta distribution

    # input requirement: (pixel_u, pixel_v, sample_num); so we make pixel_v=1
    sdf_mean_im = sdf_mean_full.unsqueeze(1)
    sdf_std_im = sdf_std_full.unsqueeze(1)

    # (pixel_u, pixel_v)
    E_d, Var_d = uncertain_render_beta_io(sdf_mean_im, sdf_std_im, sampled_ray_depth, 
                                          device=device, dtype=dtype,
                                          BACKGROUND_DEPTH=BACKGROUND_DEPTH)


    E_d = E_d.squeeze().unsqueeze(0)  # (1, points_num, )
    Var_d = Var_d.squeeze().unsqueeze(0) # (1, points_num, )

    # TODO: Try clamp
    # # make it more robust and stable
    # res_d[res_d > 0.30] = 0.30
    # res_d[res_d < -0.30] = -0.30
    # res_d = res_d.view(-1, 1, 1)

    mask_valid_depth = (E_d > 0.0) & (Var_d > 0)
    E_d_valid = E_d[mask_valid_depth]
    Var_d_valid = Var_d[mask_valid_depth]

    # calculate loss with ES/NLL w.r.t depth_obs
    depth_obs = depth_obs.to(device).unsqueeze(0)
    depth_obs_valid = depth_obs[mask_valid_depth]

    goals = depth_obs_valid
    means = E_d_valid
    variances = Var_d_valid

    if loss_type == "NLL":
        loss = ((means - goals).square() / variances + variances.log()).mean()
    elif loss_type == 'NLL_pure':
        loss = ((means - goals).square() / variances).mean()

        # print("NLL_pure loss:", loss)
        # print('log_variance:', variances.log().mean())

    elif loss_type == "normal":
        normal_loss = ( (means - goals).square() ).mean()
        loss = normal_loss
    elif loss_type == "energy_score":

        loss = loss_energy_score_batch_mu_var(means.unsqueeze(-1), 
                                              variances.unsqueeze(-1), 
                                              goals.unsqueeze(-1), M_sample=1000)
    elif loss_type == 'l1':
        loss_l1 = torch.nn.L1Loss()
        loss = loss_l1(means, goals)

    if torch.isnan(loss) or torch.isinf(loss):
        print('find invalid loss.')

    return loss


def compute_render_loss_uncertain_beta_jac(decoder, ray_directions, depth_obs,
                                       t_obj_cam, t_obj_cam_1, dPose_std, dScale_std,
                                       sampled_ray_depth, latent_vector_distribution,
                                       th=0.01, sample_num=10,
                                       loss_type='energy_score',
                                       dtype=torch.float64,
                                       BACKGROUND_DEPTH=9.0):

    # (n_rays, num_samples_per_ray, 3) = (n_rays, 1, 3) * (num_samples_per_ray, 1)
    sampled_points_cam = ray_directions[..., None, :] * sampled_ray_depth[:, None]
    # (n_rays, num_samples_per_ray, 3)
    sampled_points_obj = (sampled_points_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]
    n_rays = sampled_points_obj.shape[0]
    n_depths = sampled_ray_depth.shape[0]
    valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)

    query_points_cam = sampled_points_cam[valid_indices[0], valid_indices[1], :]
    pts_surface_cam = torch.cat((query_points_cam, torch.ones((query_points_cam.size(0), 1),
                                                     dtype=torch.float32).cuda()), dim=1)
    pts_surface_obj = torch.mm(t_obj_cam, pts_surface_cam.permute(1, 0)).permute(1, 0)[:, :3]
    x_obj = (pts_surface_cam[..., None, :3] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]

    # if too few query points, return immediately
    if pts_surface_obj.shape[0] < 10:
        return None

    # Go through decoder functions to get SDF mean
    ####################################################################
    device = latent_vector_distribution.device
    # if loss_type is normal, we do not sample and only use the mean!
    latent_codes_samples = latent_vector_distribution[:,0].unsqueeze(0)
    # (sample_num, points_num, 1)
    n_points = pts_surface_obj.shape[0]
    latent_repeat = latent_codes_samples.expand(n_points, -1)
    inputs = torch.cat([latent_repeat, pts_surface_obj], -1)
    sdf_est = decoder(inputs)
    ####################################################################

    # Jacobian related
    #############################################################################
    out_dim_ = 1
    n_ = x_obj.shape[0]
    input_x_ = x_obj.clone().detach()
    lat_vec_ = latent_vector_distribution[:,0].clone().detach()
    latent_repeat_ = lat_vec_.expand(n_, -1)
    inputs_ = torch.cat([latent_repeat_, input_x_], 1)
    inputs_ = inputs_.unsqueeze(1)  # (n, 1, in_dim)
    inputs_ = inputs_.repeat(1, out_dim_, 1)
    inputs_.requires_grad = True
    y_ = decoder(inputs_)  # (n, out_dim, out_dim)
    #y_ = torch.clamp(y_, -clamp_dist, clamp_dist)
    w_ = torch.eye(out_dim_).view(1, out_dim_, out_dim_).repeat(n_, 1, 1).cuda()
    y_.backward(w_, retain_graph=False)
    de_di = inputs_.grad.data.detach()
    # Jacobian for SDF to code
    jac_code = de_di[..., :-3]
    # Jacobian for SDF to object point
    de_dxo = de_di[..., -3:]
    # Jacobian for object point to 6D pose
    dxo_dtoc = get_points_to_pose_jacobian_se3(x_obj)
    # Jacobian for object point to scale point
    dxo_dxs = get_points_to_points_jacobian(x_obj, t_obj_cam_1)
    # Jacobian for scale point to 3D scale
    dxs_ds = get_points_to_scale_jacobian(pts_surface_cam)
    # Jacobian for SDF to 6D pose
    jac_toc = torch.bmm(de_dxo, dxo_dtoc)
    # Jacobian for SDF to 3D scale
    jac_scale = torch.bmm(de_dxo, torch.bmm(dxo_dxs, dxs_ds))
    #############################################################################
    # Full Jacobian
    jac_full = torch.cat([jac_code, jac_toc, jac_scale], -1)


    # Compute SDF uncertainty and loss
    ####################################################################################
    code_pose_dis = torch.cat([torch.exp(latent_vector_distribution[:, 1]),
                               torch.exp(dPose_std),
                               torch.exp(dScale_std)], -1)
    code_pose_dis = torch.diag(code_pose_dis)
    code_pose_dis = code_pose_dis.view(1, 73, 73).repeat(n_, 1, 1)
    sdf_unc = torch.bmm(jac_full, torch.bmm(code_pose_dis, jac_full.permute(0, 2, 1)))
    ####################################################################################
    '''
    code_dis = torch.exp(latent_vector_distribution[:, 1])
    code_dis = torch.diag(code_dis)
    code_dis = code_dis.view(1, 64, 64).repeat(n_, 1, 1)
    sdf_unc = torch.bmm(jac_code, torch.bmm(code_dis, jac_code.permute(0, 2, 1)))
    '''

    sdf_mean = sdf_est.squeeze()
    sdf_var = sdf_unc.squeeze()
    sdf_std = sdf_var.sqrt()
    # Clamp large sdf_std
    sdf_std = torch.clamp(sdf_std, 1e-12, 0.25)



    # Full dimension (n_rays, n_samples_per_ray)
    SDF_INVALID = 1.0  # for those outside of the range (CONFIRMED the same as before)
    SIGMA_INVALID = 1.0

    sdf_mean_full = torch.full((n_rays, n_depths), SDF_INVALID).cuda()
    sdf_std_full = torch.full((n_rays, n_depths), SIGMA_INVALID).cuda()
    valid_indices_x, valid_indices_y = valid_indices  # (indices on x, y dimension)
    sdf_mean_full[valid_indices_x, valid_indices_y] = sdf_mean
    sdf_std_full[valid_indices_x, valid_indices_y] = sdf_std

    # render with beta distribution

    # input requirement: (pixel_u, pixel_v, sample_num); so we make pixel_v=1
    sdf_mean_im = sdf_mean_full.unsqueeze(1)
    sdf_std_im = sdf_std_full.unsqueeze(1)

    # (pixel_u, pixel_v)
    E_d, Var_d = uncertain_render_beta_io(sdf_mean_im, sdf_std_im, sampled_ray_depth,
                                          device=device, dtype=dtype,
                                          BACKGROUND_DEPTH=BACKGROUND_DEPTH)
    print(sdf_mean_im.shape, sdf_std_im.shape, sampled_ray_depth.shape)

    E_d = E_d.squeeze().unsqueeze(0)  # (1, points_num, )
    Var_d = Var_d.squeeze().unsqueeze(0)  # (1, points_num, )

    # TODO: Try clamp
    # # make it more robust and stable
    # res_d[res_d > 0.30] = 0.30
    # res_d[res_d < -0.30] = -0.30
    # res_d = res_d.view(-1, 1, 1)

    mask_valid_depth = (E_d > 0.0) & (Var_d > 0)
    E_d_valid = E_d[mask_valid_depth]
    Var_d_valid = Var_d[mask_valid_depth]

    # calculate loss with ES/NLL w.r.t depth_obs
    depth_obs = depth_obs.to(device).unsqueeze(0)
    depth_obs_valid = depth_obs[mask_valid_depth]

    goals = depth_obs_valid
    means = E_d_valid
    variances = Var_d_valid

    print(goals.shape, means.shape, variances.shape)

    if loss_type == "NLL":
        loss = ((means - goals).square() / variances + variances.log()).mean()
    elif loss_type == 'NLL_pure':
        loss = ((means - goals).square() / variances).mean()

        # print("NLL_pure loss:", loss)
        # print('log_variance:', variances.log().mean())

    elif loss_type == "normal":
        normal_loss = ((means - goals).square()).mean()
        loss = normal_loss
    elif loss_type == "energy_score":

        loss = loss_energy_score_batch_mu_var(means.unsqueeze(-1),
                                              variances.unsqueeze(-1),
                                              goals.unsqueeze(-1), M_sample=1000)
    elif loss_type == 'l1':
        loss_l1 = torch.nn.L1Loss()
        loss = loss_l1(means, goals)

    if torch.isnan(loss) or torch.isinf(loss):
        print('find invalid loss.')

    return loss


def compute_render_loss_uncertain_jac_multiview(decoder, ray_directions_mv, depth_obs_mv,
                                       t_obj_cam_mv, t_obj_cam_1_mv, dPose_std, dScale_std,
                                       sampled_ray_depth_mv, latent_vector_distribution,
                                       th=0.01, sample_num=10,
                                       loss_type='energy_score',
                                       dtype=torch.float64,
                                    #    BACKGROUND_DEPTH=9.0,
                                        background_depth_mv=np.array([9.0]),
                                       render_2d_K=400,
                                       render_2d_calibrate_C=1.0,
                                       render_2d_const_a=None):
    
    n_depths = sampled_ray_depth_mv[0].shape[0]
    sdf_mean_im_stacked = torch.empty((0, n_depths),dtype=dtype).cuda()
    sdf_std_im_stacked = torch.empty((0, n_depths),dtype=dtype).cuda()
    sampled_ray_depth_stacked = torch.empty((0),dtype=dtype).cuda()
    depth_obs_stacked = torch.empty((0),dtype=dtype).cuda()
    background_stacked = torch.empty((0),dtype=dtype).cuda()

    n_views = len(ray_directions_mv)

    for i in range(n_views):

        # (n_rays, num_samples_per_ray, 3) = (n_rays, 1, 3) * (num_samples_per_ray, 1)
        sampled_points_cam = ray_directions_mv[i][..., None, :] * sampled_ray_depth_mv[i][:, None]

        # (n_rays, num_samples_per_ray, 3)
        sampled_points_obj = (sampled_points_cam[..., None, :] * t_obj_cam_mv[i][:3, :3]).sum(-1) + t_obj_cam_mv[i][:3, 3]
        n_rays = sampled_points_obj.shape[0]
        n_depths = sampled_ray_depth_mv[i].shape[0]
        valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)

        query_points_cam = sampled_points_cam[valid_indices[0], valid_indices[1], :]
        pts_surface_cam = torch.cat((query_points_cam, torch.ones((query_points_cam.size(0), 1),
                                                        dtype=torch.float32, device=torch.device("cuda"))), dim=1)
        pts_surface_obj = torch.mm(t_obj_cam_mv[i], pts_surface_cam.permute(1, 0)).permute(1, 0)[:, :3]
        x_obj = (pts_surface_cam[..., None, :3] * t_obj_cam_mv[i][:3, :3]).sum(-1) + t_obj_cam_mv[i][:3, 3]

        # if too few query points, return immediately
        if pts_surface_obj.shape[0] < 10:
            return None

        # Go through decoder functions to get SDF mean
        ####################################################################
        # if loss_type is normal, we do not sample and only use the mean!
        latent_codes_samples = latent_vector_distribution[:,0].unsqueeze(0)
        # (sample_num, points_num, 1)
        n_points = pts_surface_obj.shape[0]
        latent_repeat = latent_codes_samples.expand(n_points, -1)
        inputs = torch.cat([latent_repeat, pts_surface_obj], -1)
        sdf_est = decoder(inputs)
        ####################################################################

        # Jacobian related
        #############################################################################
        out_dim_ = 1
        n_ = x_obj.shape[0]
        input_x_ = x_obj.clone().detach()
        lat_vec_ = latent_vector_distribution[:,0].clone().detach()
        latent_repeat_ = lat_vec_.expand(n_, -1)
        inputs_ = torch.cat([latent_repeat_, input_x_], 1)
        inputs_ = inputs_.unsqueeze(1)  # (n, 1, in_dim)
        inputs_ = inputs_.repeat(1, out_dim_, 1)
        inputs_.requires_grad = True
        y_ = decoder(inputs_)  # (n, out_dim, out_dim)
        #y_ = torch.clamp(y_, -clamp_dist, clamp_dist)
        w_ = torch.eye(out_dim_).view(1, out_dim_, out_dim_).repeat(n_, 1, 1).cuda()
        y_.backward(w_, retain_graph=False)
        de_di = inputs_.grad.data.detach()
        # Jacobian for SDF to code
        jac_code = de_di[..., :-3]
        # Jacobian for SDF to object point
        de_dxo = de_di[..., -3:]
        # Jacobian for object point to 6D pose
        dxo_dtoc = get_points_to_pose_jacobian_se3(x_obj)
        # Jacobian for object point to scale point
        dxo_dxs = get_points_to_points_jacobian(x_obj, t_obj_cam_1_mv[i])
        # Jacobian for scale point to 3D scale
        dxs_ds = get_points_to_scale_jacobian(pts_surface_cam)
        # Jacobian for SDF to 6D pose
        jac_toc = torch.bmm(de_dxo, dxo_dtoc)
        # Jacobian for SDF to 3D scale
        jac_scale = torch.bmm(de_dxo, torch.bmm(dxo_dxs, dxs_ds))
        #############################################################################
        # Full Jacobian
        jac_full = torch.cat([jac_code, jac_toc, jac_scale], -1)

        # Compute SDF uncertainty and loss
        ####################################################################################
        code_pose_dis = torch.cat([torch.exp(latent_vector_distribution[:, 1]),
                                torch.exp(dPose_std),
                                torch.exp(dScale_std)], -1)
        code_pose_dis = torch.diag(code_pose_dis)
        code_pose_dis = code_pose_dis.view(1, 73, 73).repeat(n_, 1, 1)
        sdf_unc = torch.bmm(jac_full, torch.bmm(code_pose_dis, jac_full.permute(0, 2, 1)))
        ####################################################################################


        sdf_mean = sdf_est.squeeze()

        open_clamp = True

        if open_clamp:
            '''
            Note: Please set 0.25 for KITTI. 0.5 for ScanNet (to produce paper result)
            '''
            sdf_std = torch.clamp(sdf_unc.squeeze().sqrt(), 1e-12, 0.25)   # TODO: Variance cut to 0.25
        else:
            sdf_std = sdf_unc.squeeze().sqrt()

        # Full dimension (n_rays, n_samples_per_ray)
        SDF_INVALID = 1.0  # for those outside of the range (CONFIRMED the same as before)
        SIGMA_INVALID = 1.0

        sdf_mean_full = torch.full((n_rays, n_depths), SDF_INVALID).cuda()
        sdf_std_full = torch.full((n_rays, n_depths), SIGMA_INVALID).cuda()
        valid_indices_x, valid_indices_y = valid_indices  # (indices on x, y dimension)
        sdf_mean_full[valid_indices_x, valid_indices_y] = sdf_mean
        sdf_std_full[valid_indices_x, valid_indices_y] = sdf_std

        sdf_mean_im_stacked = torch.vstack((sdf_mean_im_stacked, sdf_mean_full))
        sdf_std_im_stacked = torch.vstack((sdf_std_im_stacked, sdf_std_full))
        sampled_ray_depth_stacked = torch.hstack((sampled_ray_depth_stacked, sampled_ray_depth_mv[i]))
        depth_obs_stacked = torch.hstack((depth_obs_stacked, depth_obs_mv[i]))
        background_view = torch.tensor(np.tile(background_depth_mv[i], [1, n_rays])).cuda()
        background_stacked = torch.hstack((background_stacked, background_view))
    # render with beta distribution

    # input requirement: (pixel_u, pixel_v, sample_num); so we make pixel_v=1
    sdf_mean_im_stacked = sdf_mean_im_stacked.unsqueeze(1)
    sdf_std_im_stacked = sdf_std_im_stacked.unsqueeze(1)

    # (pixel_u, pixel_v)
    E_d, Var_d = uncertain_render_beta_io(sdf_mean_im_stacked, sdf_std_im_stacked, sampled_ray_depth_stacked,
                                          dtype=dtype,
                                          BACKGROUND_DEPTH=background_stacked, k=render_2d_K,
                                          const_a = render_2d_const_a)

    E_d = E_d.squeeze().unsqueeze(0)  # (1, points_num, )
    Var_d = Var_d.squeeze().unsqueeze(0)  # (1, points_num, )

    # TODO: Try clamp
    # # make it more robust and stable
    # res_d[res_d > 0.30] = 0.30
    # res_d[res_d < -0.30] = -0.30
    # res_d = res_d.view(-1, 1, 1)

    mask_valid_depth = (E_d > 0.0) & (Var_d > 0)
    E_d_valid = E_d[mask_valid_depth]
    Var_d_valid = Var_d[mask_valid_depth]

    # calculate loss with ES/NLL w.r.t depth_obs
    depth_obs_valid = depth_obs_stacked[mask_valid_depth.reshape(-1)]

    goals = depth_obs_valid
    means = E_d_valid
    variances = Var_d_valid

    '''
    TEMP DEBUG: Save the distribution of SDF values for every 10 iterations
    '''
    debug_save_med_result = False
    if debug_save_med_result:
        # result_list = []

        global temp_output_list

        output = {
            'sdf_mean': sdf_mean_im_stacked.detach().cpu(),
            'sdf_std': sdf_std_im_stacked.detach().cpu(),
            'sampled_ray_depth_stacked': sampled_ray_depth_stacked.detach().cpu(),
            'BACKGROUND_DEPTH': BACKGROUND_DEPTH,
            'render_2d_K': render_2d_K,
            'E_d': E_d.detach().cpu(),
            'Var_d': Var_d.detach().cpu(),
            'iter': len(temp_output_list)
        }

        temp_output_list.append(output)

        # check if iterations == 200
        if len(temp_output_list) == 200:
            print('final iter!')

            # skip every 10
            temp_output_list_skip = temp_output_list[::5]
            # save to local disk with pickle
            import pickle
            with open('temp_output_list_skip.pkl', 'wb') as f:
                pickle.dump(temp_output_list_skip, f)


    '''
     calibration process
    '''
    variances = variances * render_2d_calibrate_C

    if loss_type == "NLL":
        loss = ((means - goals).square() / variances + variances.log()).mean()
    elif loss_type == 'NLL_pure':
        loss = ((means - goals).square() / variances).mean()

    elif loss_type == "normal":
        normal_loss = ((means - goals).square()).mean()
        loss = normal_loss
    elif loss_type == "energy_score":

        loss = loss_energy_score_batch_mu_var(means.unsqueeze(-1),
                                              variances.unsqueeze(-1),
                                              goals.unsqueeze(-1), M_sample=1000)
    elif loss_type == 'l1':
        loss_l1 = torch.nn.L1Loss()
        loss = loss_l1(means, goals)

    if torch.isnan(loss) or torch.isinf(loss):
        print('find invalid loss.')

    return loss




def compute_render_loss_uncertain(decoder, ray_directions, depth_obs, t_obj_cam, sampled_ray_depth, latent_vector_distribution, th=0.01, sample_num=10):
    """
    Method: Sample multiple times to get multiple depth map and calculate the mean and variance.

    :param decoder: DeepSDF decoder
    :param ray_directions: (N, 3) under camera coordinate
    :param depth_obs: (N,) observed depth values for foreground pixels, 1.1 * d_max for background pixels
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param sampled_ray_depth: (M,) linspace between d_min and d_max
    :param latent_vector: shape code
    :param th: cut-off threshold for converting SDF to occupancy
    :return: Jacobian wrt pose (K, 1, 7), Jacobian wrt shape code (K, 1, code_len), error residuals (K, 1, 1)
    Note that K is the number of points that have non-zero Jacobians out of a total of N * M points
    """
    
    latent_codes_samples = sample_with_reparameterization(latent_vector_distribution, sample_num, logvar_to_sigma=True)
    
    error_list = []
    for latent_vector in latent_codes_samples:
        error = compute_render_loss_nojac(decoder, ray_directions, depth_obs, t_obj_cam, sampled_ray_depth, latent_vector, th=0.01)
        
        # add all error from each sampling
        error_list.append(error)
        
    # Cat to the first dimension
    res_d = torch.cat(error_list, dim=0)

    return res_d

def compute_rotation_loss_sim3(t_obj_cam):
    """
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :return: Jacobian and residual of rotation regularization term
    """
    # E_rot = 1 - ry * ng
    t_cam_obj = torch.inverse(t_obj_cam)
    r_co = t_cam_obj[:3, :3]
    scale = torch.det(r_co) ** (1 / 3)
    r_co /= scale
    r_oc = torch.inverse(r_co)

    ey = torch.tensor([0., 1., 0.])
    ng = torch.tensor([0., -1., 0.])
    ry = torch.mv(r_co, ey)
    res_rot = 1. - torch.dot(ry, ng)
    if res_rot < 1e-7:
        return torch.zeros(7), 0.

    J_rot = torch.cross(torch.mv(r_oc, ng), ey)
    J_sim3 = torch.zeros(7)
    J_sim3[3:6] = J_rot

    return J_sim3, res_rot

def test_distribution_of_vector(v, save_name, type='hist', title=None):
    '''
    @v: (N, 1)
    '''

    # plot a bar chart
    v = v.detach().squeeze().cpu().numpy()
    # import
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    if type == 'hist':
        plt.hist(v, bins=100)
    else:
        plt.bar(range(len(v)), v)
    if title is not None:
        plt.title(title)
    plt.xlabel('value')
    plt.ylabel('count')
    plt.savefig(save_name)

    plt.close()