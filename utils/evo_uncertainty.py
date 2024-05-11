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

'''
A script to analyze / visualize uncertainty.
'''

import os,sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__), '..'))

import torch

from reconstruct.loss_utils import decode_sdf, get_batch_sdf_jacobian, get_points_to_pose_jacobian_se3, get_batch_sdf_values,\
    get_points_to_pose_jacobian_sim3, get_points_to_points_jacobian, get_points_to_scale_jacobian, sdf_to_occupancy

import time

import numpy as np

from uncertainty.energy_score import loss_energy_score_batch_mu_var
from reconstruct.utils import color_table, set_view, get_configs, get_decoder

from utils import SE3

import open3d as o3d

from run_system import visualize_o3d_ins_to_image


def propagate_pose_code_uncertainty_to_sdf(decoder, data_surface_world, latent_vector_distribution, 
                                           T_ow, T_ow_1, dPose_std, dScale_std,
                                            clamp_dist=0.1,
                                            points_sample_each_iter=None):
    '''
    @loss_type: if normal, we only use the origin dsp-slam method; if energy_score, we propagate uncertainty with jacobian

    @points_sample_each_iter: Update for Scannet; Sample N points for each iteration; None: Consider all.
    '''
    time_0 = time.time()

    # update: consider the case where pts_surface_cam contain gt sdf values
    if points_sample_each_iter is None or (len(data_surface_world) < points_sample_each_iter):
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

    b_use_uncertainty = True
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

    # sdf_values_ob = sdf_values_ob.to(device)
    # goals = sdf_values_ob
       
    # normal_loss = ((sdf_mean - sdf_values_ob).square()).mean(0)
    # if loss_type == 'energy_score':
    #     sdf_loss = loss_energy_score_batch_mu_var(sdf_mean, sdf_var, goals, M_sample=1000)
    # elif loss_type == 'normal':
    #     sdf_loss = normal_loss
    # elif loss_type == 'nll':
    #     # elif loss_type == 'NLL_pure':
    #     #     loss = ((means - goals).square() / variances).mean()
    #     sdf_loss = ((sdf_mean - sdf_values_ob).square() / sdf_var + sdf_var.log()).mean(0)

    detail = {
        'pts_surface_obj': pts_surface_obj.detach().cpu()
    }

    return sdf_mean, sdf_var, detail



def calculate_es_nll_3d(decoder, data_surface_world, latent_vector_distribution, 
                                           T_ow, T_ow_1, dPose_std, dScale_std, device='cuda', th=None):
    '''
    Calculate the negative log likelihood or Energy Score of the estimated shape.

    Input: gt_pts_world: sampled points from GT mesh in world.

    @ gt_t_world_obj: (4,4) np.ndarray, estimated pose, with uncertainty
    @ gt_t_world_obj_std: uncertainty of pose

    @ latent code,
    @ sigma code
    '''

    # make (N,3) -> (N,4)
    data_surface_world = torch.cat([data_surface_world.to(device), torch.ones(data_surface_world.shape[0], 1).to(device)], -1)
    latent_vector_distribution = latent_vector_distribution.to(device)
    T_ow = T_ow.to(device)
    T_ow_1 = torch.from_numpy(T_ow_1).to(device)
    dPose_std = dPose_std.to(device)
    dScale_std = dScale_std.to(device)

    gt_points_sdf, \
    gt_points_sdf_var, detail = propagate_pose_code_uncertainty_to_sdf(decoder, data_surface_world, latent_vector_distribution, 
                                           T_ow, T_ow_1, dPose_std, dScale_std)

    '''
    th
    '''
    if th is not None:
        # only consider sdf points inside th
        mask = (gt_points_sdf.abs() < th)
        gt_points_sdf = gt_points_sdf[mask].unsqueeze(-1)
        gt_points_sdf_var = gt_points_sdf_var[mask].unsqueeze(-1)

        detail['pts_surface_obj'] = detail['pts_surface_obj'][mask.detach().cpu().squeeze()].unsqueeze(-1)

        # check if there are enough points
        if len(gt_points_sdf) < 1:
            print('no points inside th')
            return None, None, None


    # Step 3: calcualte the negative log likelihood of the SDF
    # negative log likelihood of the SDF
    goals = torch.zeros(gt_points_sdf.shape).to(device)
    means = gt_points_sdf
    variances = gt_points_sdf_var

    nll_loss = ((means - goals).square() / variances + variances.log()).mean()
    es_loss = loss_energy_score_batch_mu_var(means, 
                                            variances, 
                                            goals, M_sample=1000)    
    normal_loss = ((means - goals).square()).mean()

    # print('nll_loss: ', nll_loss)
    # print('es_loss: ', es_loss)
    # print('normal_loss: ', normal_loss)

    loss = {
        'nll': nll_loss.detach().cpu().numpy(),
        'es': es_loss.detach().cpu().numpy(),
        'normal': normal_loss.detach().cpu().numpy()
    }

    return loss, gt_points_sdf, gt_points_sdf_var, detail

def calculate_pearsonr(sdfs_abs, sigmas, bins=None):
    if bins is not None and bins > 0:
        sdf_abs_bins = torch.linspace(0, sdfs_abs.max(), bins)
        valid_sdf_list = []
        mean_sigma_list = []

        for i in range(len(sdf_abs_bins)-1):
            mask = (sdfs_abs >= sdf_abs_bins[i]) & (sdfs_abs < sdf_abs_bins[i+1])
            # make sure there are enough data
            if torch.sum(mask) < 1:
                continue
            mean_sigma_list.append(torch.mean(sigmas[mask]))
            valid_sdf_list.append(sdf_abs_bins[i])
        bins_mean_sigma = torch.stack(mean_sigma_list)
        bins_valid_sdf = torch.stack(valid_sdf_list)
        bins_input_pearson = torch.stack([bins_valid_sdf, bins_mean_sigma])
        R = torch.corrcoef(bins_input_pearson)
        r = R[0,1]
    else:
        input_pearson = torch.cat([sdfs_abs, sigmas], -1).detach().cpu().transpose(0,1)
        R = torch.corrcoef(input_pearson)
        # get one value form the 2x2 matrix
        r = R[0,1]

    return r

def reconstruct_sdf_error_uncertainty_mesh_pair_improve(decoder, latent_code, 
                                                        T_ow, T_ow_1, T_oc_std, scale_std, pts_obj, 
                                                        sdfs_abs, sigmas, save_dir, vis=None):
    '''
    An improved version to calculate both error and uncertainty.

    Consider vertices of the reconstructed mesh, use the closest distance to gt sample points as errors.

    Use the vertices in obj, to the world, then calculate pose&code propagate uncertainty, as sigmas.
    '''
    
    # reconstruct mesh
    # mesh = decode_sdf(decoder, latent_code, N=256, max_batch=32 ** 3, offset=0., scale=1., device='cuda')
    code = latent_code[:,0]

    # from reconstruct.optimizer import Optimizer, MeshExtractor
    from uncertainty.optimizer_uncertainty import MeshExtractorUncertain
    mesh_extractor = MeshExtractorUncertain(decoder, voxels_dim=64)

    mesh = mesh_extractor.extract_mesh_from_code(code)

    vertices = mesh.vertices  # 5k points  [-1,1]

    errors = []

    # for those vertices, (~1K points), find the closest points in sdfs_abs, sigmas
    # find closest indices to pts_obj
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_obj.detach().cpu().squeeze().numpy())
    dist, idx = tree.query(vertices, k=1)

    # get the sdf error and uncertainty
    sdf_error = dist  # matched dist is just the errors of the vertices!

    # begin calculating uncertainty
    # transform vertices vert_o (N,3) into world, with T_ow; vert_w = T_wo @ vert_o
    T_wo = T_ow.inverse()
    vertices = torch.from_numpy(vertices).cuda()
    vertices_world_homo = torch.mm(T_wo, torch.cat([vertices, torch.ones(vertices.shape[0], 1).cuda()], -1).transpose(0,1)).transpose(0,1)

    # propagate uncertainty
    T_ow_1 = torch.from_numpy(T_ow_1).cuda()
    T_oc_std = T_oc_std.cuda()
    scale_std = scale_std.cuda()

    gt_points_sdf, \
    gt_points_sdf_var, detail = propagate_pose_code_uncertainty_to_sdf(decoder, vertices_world_homo, latent_code.cuda(), 
                                           T_ow, T_ow_1, T_oc_std, scale_std)

    # Check points sdf should be zero!
    # print('gt_points_sdf: ', gt_points_sdf.sum())
    # begin visualize mesh, with error, and gt_points_sdf_var

    sdf_uncertainty = gt_points_sdf_var.sqrt().squeeze().detach().cpu()
    
    # colorize the mesh
    # TODO
    # relative color
    # normalize them to ignore scale
    color_error = mesh_extractor.sigma_vec_to_color_vec(sdf_error, vis_abs_uncer=False)
    color_uncertainty = mesh_extractor.sigma_vec_to_color_vec(sdf_uncertainty, vis_abs_uncer=False)

    # construct open3d mesh
    vertices = vertices.detach().cpu().numpy()
    # import open3d as o3d
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(color_error)

    mesh_o3d.compute_vertex_normals()

    # save
    o3d.io.write_triangle_mesh(f'{save_dir}/new_mesh_error.ply', mesh_o3d)
    save_im_name = os.path.join(save_dir, 'new_mesh_error_vn.png')
    visualize_o3d_ins_to_image([mesh_o3d], './view_file_deepsdf.json', save_im_name, vis)
    

    # colorize the mesh
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(color_uncertainty)
    mesh_o3d.compute_vertex_normals()

    # save
    o3d.io.write_triangle_mesh(f'{save_dir}/new_mesh_uncertainty.ply', mesh_o3d)
    save_im_name = os.path.join(save_dir, 'new_mesh_uncertainty_vn.png')
    visualize_o3d_ins_to_image([mesh_o3d], './view_file_deepsdf.json', save_im_name, vis)
    
    # visualize vertices as point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_point_cloud(f'{save_dir}/pts_vertices.ply', pcd)
    save_im_name = os.path.join(save_dir, 'pts_vertices.png')
    visualize_o3d_ins_to_image([pcd], './view_file_deepsdf.json', save_im_name, vis)
    
    # save GT Points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_obj.detach().cpu().numpy())
    o3d.io.write_point_cloud(f'{save_dir}/pts_obj_gt_samples.ply', pcd)
    save_im_name = os.path.join(save_dir, 'pts_obj_gt_samples.png')
    visualize_o3d_ins_to_image([pcd], './view_file_deepsdf.json', save_im_name, vis)
    


def reconstruct_sdf_error_uncertainty_mesh_pair(decoder, latent_code, pts_obj, sdfs_abs, sigmas, save_dir):
    '''
    Method 1: 
        sample vertices, and find the closest points in sdfs_abs, sigmas

    Limitations:
        when the nearest points do not exist, there are a lot of errors.

    Reconstruct a mesh, and colorize it with sdf error and uncertainty.
    
    Two meshes output.
    '''

    # reconstruct mesh
    # mesh = decode_sdf(decoder, latent_code, N=256, max_batch=32 ** 3, offset=0., scale=1., device='cuda')
    code = latent_code[:,0]

    # from reconstruct.optimizer import Optimizer, MeshExtractor
    from uncertainty.optimizer_uncertainty import MeshExtractorUncertain
    mesh_extractor = MeshExtractorUncertain(decoder, voxels_dim=64)

    mesh = mesh_extractor.extract_mesh_from_code(code)

    vertices = mesh.vertices  # 5k points  [-1,1]

    # for those vertices, (~1K points), find the closest points in sdfs_abs, sigmas
    # find closest indices to pts_obj
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_obj.detach().cpu().numpy())
    dist, idx = tree.query(vertices, k=1)

    # get the sdf error and uncertainty
    sdf_error = sdfs_abs[idx].squeeze()
    sdf_uncertainty = sigmas[idx].squeeze()

    # colorize the mesh
    # TODO
    # relative color
    # normalize them to ignore scale
    color_error = mesh_extractor.sigma_vec_to_color_vec(sdf_error, vis_abs_uncer=False)
    color_uncertainty = mesh_extractor.sigma_vec_to_color_vec(sdf_uncertainty, vis_abs_uncer=False)

    # construct open3d mesh
    # import open3d as o3d
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(color_error)

    # save
    o3d.io.write_triangle_mesh(f'{save_dir}/mesh_error.ply', mesh_o3d)

    # colorize the mesh
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(color_uncertainty)

    # save
    o3d.io.write_triangle_mesh(f'{save_dir}/mesh_uncertainty.ply', mesh_o3d)

    # visualize vertices as point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_point_cloud(f'{save_dir}/pts_vertices.ply', pcd)

    # save GT Points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_obj.detach().cpu().numpy())
    o3d.io.write_point_cloud(f'{save_dir}/pts_obj_gt_samples.ply', pcd)


def evaluate_instance(obj_data, pts1_sampled_world, decoder=None, 
                      plot_calib=True, save_dir='./output', th=None,
                      vis_mesh=False, vis=None):
    '''
    Calcualte a calibration plot based on the uncertainty of the estimated shape.
    '''
    if decoder is None:
        # load decoder
        config_dir = 'configs/config_redwood_01053.json'
        configs = get_configs(config_dir)
        decoder = get_decoder(configs)

    #########
    intermediate = obj_data['intermediate']

    step = -1

    ##########################
    # Load the Input Data
    ##########################
    # load final output std and pose and code
    latent_code = intermediate['code'][step]  # (64,2)
    # change sigma to log(var)
    latent_code[:,1] = latent_code[:,1].square().log()

    T_oc = intermediate['T_oc'][step] # (4,4)   # with scale  # scale in world coordinate
    T_oc_std = intermediate['T_oc_std'][step] #(6,)
    scale_std = intermediate['scale_std'][step] #(3,)

    T_cam_obj = torch.from_numpy(obj_data['t_cam_obj'])
    T_world_obj = torch.from_numpy(obj_data['t_world_obj'])
    T_wc_vis = T_world_obj @ T_cam_obj.inverse()
    T_wc_vis = T_wc_vis.cuda()

    # Get GT Points in world
    gt_points_world = pts1_sampled_world

    T_oc = T_oc.cuda()
    T_ow = T_oc @ T_wc_vis.inverse()  # with scale
    t_ow, q_ow, s_ow  = SE3.decompose_mat4(T_ow.cpu().numpy())
    T_ow_1 = SE3.compose_mat4(t_ow, q_ow, np.ones(3)).astype(np.float32)  # without scale

    ##########################


    # Sainity Check:  Input the local points, and should get the same loss recorded in loss curve
    '''
    Note if th is not None, the sdfs and vars are only inside th.
    '''
    loss, sdfs, vars, detail = calculate_es_nll_3d(decoder, gt_points_world, latent_code, 
                                        T_ow, T_ow_1, T_oc_std, scale_std, th=th)

    sdfs_abs = sdfs.abs().detach().cpu()
    sigmas = torch.sqrt(vars).detach().cpu()

    if loss is not None and plot_calib:
        # plot calib mat from sdfs, vars
        make_calibration_plot(sdfs,vars,save_dir=save_dir)


    # sdfs: (N,1) ; vars: (N,1)
    # combine into one vector
    if loss is not None:
        r = calculate_pearsonr(sdfs_abs, sigmas)
        print('pearsonr: ', r)
        loss['pearson'] = r.item()

        # use bins to calculate the mean
        r = calculate_pearsonr(sdfs_abs, sigmas, bins=100)
        print('pearsonr bins: ', r)
        loss['pearson_bins'] = r.item()

        # if also reconstruct mesh with color for sdf error and uncertainty
        if vis_mesh:
            # pts_surface_obj is from the valid points
            # reconstruct_sdf_error_uncertainty_mesh_pair(decoder, latent_code, detail['pts_surface_obj'], sdfs_abs, sigmas, save_dir)

            reconstruct_sdf_error_uncertainty_mesh_pair_improve(decoder, latent_code, 
                                                                T_ow, T_ow_1, T_oc_std, scale_std,
                                                                detail['pts_surface_obj'], sdfs_abs, sigmas, save_dir,
                                                                vis=vis)

    return loss

def make_calibration_plot(sdfs, vars, save_dir='./output'):
    '''
    Take each sdf value as independent gaussian, plot it
    '''

    n = sdfs.shape[0]

    # to numpy
    sdfs = sdfs.detach().cpu().numpy()
    vars = vars.detach().cpu().numpy()

    sigmas = np.sqrt(vars)

    sdfs_abs = np.abs(sdfs)

    # plot, X axis as abs of sdf, Y axis as sigma
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(sdfs, sigmas, s=1)
    plt.xlabel('sdf')
    plt.ylabel('sigma')
    plt.savefig(f'{save_dir}/calib.png')

    plt.figure()
    plt.scatter(sdfs_abs, sigmas, s=1)
    plt.xlabel('sdf_abs')
    plt.ylabel('sigma')
    plt.savefig(f'{save_dir}/calib_abs.png')


    from deep_sdf.metrics.uncertain import draw_pdf_calibration_plot
    draw_pdf_calibration_plot(sdfs, sigmas, save_dir=f'{save_dir}/calib_pdf.png')

    draw_pdf_calibration_plot(sdfs_abs, sigmas, save_dir=f'{save_dir}/calib_pdf_abs.png')

    plt.figure()
    # Bins
    N_bins = 100
    sdf_abs_bins = np.linspace(0, sdfs_abs.max(), N_bins)
    # print('ob_ratio_bins:', ob_ratio_bins)
    # calculate the mean and std of iou_list
    valid_sdf_list = []
    mean_sigma_list = []
    std_sigma_list = []
    for i in range(len(sdf_abs_bins)-1):
        mask = (sdfs_abs >= sdf_abs_bins[i]) & (sdfs_abs < sdf_abs_bins[i+1])
        # make sure there are enough data
        if np.sum(mask) < 1:
            continue
        mean_sigma_list.append(np.mean(sigmas[mask]))
        std_sigma_list.append(np.std(sigmas[mask]))
        valid_sdf_list.append(sdf_abs_bins[i])
    # print('mean_iou_list:', mean_iou_list)
    # print('std_iou_list:', std_iou_list)
    plt.errorbar(valid_sdf_list, mean_sigma_list, yerr=std_sigma_list, fmt='o')
    plt.xlabel('Abs SDF Error')
    plt.ylabel('SDF Sigma')
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.title('bins statistics for error vs sigma')
    plt.savefig(os.path.join(save_dir, 'bins.png'))

    # make X,Y lim equal
    max_lim = max(plt.xlim()[1], plt.ylim()[1])
    plt.xlim(0, max_lim)
    plt.ylim(0, max_lim)
    plt.savefig(os.path.join(save_dir, 'bins_equal.png'))

    '''
    For debug: Even saving the origin file of sdfs and vars
    '''
    print('save sdfs and sigmas!!!' * 10)
    np.save(f'{save_dir}/sdfs.npy', sdfs)
    np.save(f'{save_dir}/sigmas.npy', sigmas)

    #close all
    plt.close('all')

def evaluate_instance_sainity_check(obj_data, pts1_sampled_world):
    '''
    Go through all the iterations, and calculate the ES loss again.
    Compared with the logged ones, 
    to prove the inputs and the function are the same.

    Note: the sampled points are always the last frame, so there is little difference in the plot.
    '''
    # latent_code = obj_data['latent_code']
    
    # t_world_obj = obj_data['t_world_obj']


    # load decoder
    config_dir = 'configs/config_redwood_01053.json'
    configs = get_configs(config_dir)
    decoder = get_decoder(configs)

    #########
    intermediate = obj_data['intermediate']

    step = 0
    sdf_steps = []
    normal_steps = []

    from tqdm import tqdm
    for step in tqdm(range(len(intermediate['code']))):
        # load final output std and pose and code
        latent_code = intermediate['code'][step]  # (64,2)
        # change sigma to log(var)
        latent_code[:,1] = latent_code[:,1].square().log()

        T_oc = intermediate['T_oc'][step] # (4,4)   # with scale  # scale in world coordinate
        T_oc_std = intermediate['T_oc_std'][step] #(6,)
        scale_std = intermediate['scale_std'][step] #(3,)

        T_cam_obj = torch.from_numpy(obj_data['t_cam_obj'])
        T_world_obj = torch.from_numpy(obj_data['t_world_obj'])
        T_wc_vis = T_world_obj @ T_cam_obj.inverse()

        # Get GT Points in world
        gt_points_world = pts1_sampled_world

        # DEBUG: Set gt_points_world as local observation points
        pts_local = obj_data['pts_local']  # change from camera view into world view
        debug = True
        if debug:
            T_wc_vis = T_wc_vis.cuda()
            ob_points_world = pts_local @ T_wc_vis[:3,:3].T + T_wc_vis[:3,3]
            gt_points_world = ob_points_world

            # loss_ob = obj_data['sdf_loss_list'][step]
            # normal_loss_ob = obj_data['normal_loss_list'][step]
            # print('converged sdf_loss_ob: ', loss_ob)
            # print('converged normal_loss_ob: ', normal_loss_ob)

        # TODO: Check how to get T_ow and T_ow_1 from T_wc_vis, and T_oc
        T_oc = T_oc.cuda()
        T_ow = T_oc @ T_wc_vis.inverse()  # with scale
        t_ow, q_ow, s_ow  = SE3.decompose_mat4(T_ow.cpu().numpy())
        T_ow_1 = SE3.compose_mat4(t_ow, q_ow, np.ones(3)).astype(np.float32)  # without scale

        # Sainity Check:  Input the local points, and should get the same loss recorded in loss curve
        # with no grad
        # with torch.no_grad():
        nll, es, normal, detail = calculate_es_nll_3d(decoder, gt_points_world, latent_code, 
                                            T_ow, T_ow_1, T_oc_std, scale_std)

        # print('es: ', es)
        # print('nll: ', nll)
        # print('normal: ', normal)

        sdf_steps.append(es.detach().cpu().numpy())
        normal_steps.append(normal.detach().cpu().numpy())

    # plot, headless; compare with logged ones
    logged_es = obj_data['sdf_loss_list']
    logged_normal = obj_data['normal_loss_list']

    import matplotlib.pyplot as plt
    plt.figure()
    # two subplots
    plt.subplot(2, 1, 1)
    plt.plot(sdf_steps, label='es')
    plt.plot(logged_es, label='logged_es')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(normal_steps, label='normal')
    plt.plot(logged_normal, label='logged_normal')

    plt.legend()
    plt.savefig('output/debug/uncertain_3d/es_normal.png')
    

    # origin loss

    sainity_check = True
    if sainity_check:
        input_dict= torch.load('output/debug/uncertain_3d/data_ck.pt')

        print("checking")


def debug():
    # load a saved converged result.

    save_data_path = 'output/local/result_dataset_save.pth'

    data = torch.load(save_data_path)

    # load the data
    scene_name = 'scene0598_00'
    ins_id = 5
    frame_id = 0

    obj_data = data['results'][0]['results'][0]['results'][0]['results'][0]

    # get gt data
    from utils.scannet_subset import ScanNetSubset
    import numpy as np
    device = 'cuda'
    dataset_sub = ScanNetSubset('data/scannet', scene_name, ins_id)

    # get gt mesh points in world
    n_sample_pts = 10000
    gt_mesh = dataset_sub.get_gt_mesh()

    pts1_sampled = gt_mesh.sample_points_uniformly(number_of_points=n_sample_pts)
    pts1_sampled = torch.from_numpy(np.asarray(pts1_sampled.points)).float().unsqueeze(0)

    # to gpu
    gt_t_world_obj = dataset_sub.get_gt_obj_pose_t_world_obj()
    gt_t_world_obj = torch.from_numpy(gt_t_world_obj).float()

    # transform points, and bbox to world (N, 3) @ (4,4) -> (N, 3)
    pts1_sampled_world = pts1_sampled.squeeze(0) @ gt_t_world_obj[:3,:3].T + gt_t_world_obj[:3,3]

    # Test one object
    evaluate_instance(obj_data, pts1_sampled_world)

def io_evaluate_scene_result(scene_recon_result, condition='all', category='chair', save_dir='./output', th=None,
                             cases_list=None, vis_mesh=False, sequence_dir='data/scannet', decoder=None, mask_path_root=None):
    '''
    Given an input scene reconstruction output, evaluate all the instances met the condition.

    @ scene_recon_result: output of a run

    @ condition: all / success

    @ cases_list: if specified, only consider the objects defined by (scene_name, obj_id, frame_id) in the list.
    
    '''

    from utils.scannet_subset import ScanNetSubset
    import numpy as np
    device = 'cuda'

    n_sample_pts = 10000

    # load decoder;  TODO: consider other categories
    if decoder is None:
        if category == 'chair':
            config_dir = 'configs/config_scannet.json'
            configs = get_configs(config_dir)
            decoder = get_decoder(configs)
        elif category == 'table':
            config_dir = 'configs/config_scannet_table.json'
            configs = get_configs(config_dir)
            decoder = get_decoder(configs)
        else:
            raise NotImplementedError

    from tqdm import tqdm

    evo_result_list = []

    fail_cases = []

    '''
    Create an open3d visualizer
    '''
    if vis_mesh:
        # import open3d as o3d
        vis = o3d.visualization.Visualizer()
        # size 600, 600
        vis.create_window(width=600, height=600)
    else:
        vis = None

    n_all_data = 0
    for scene_result in tqdm(scene_recon_result['results'], desc='scene'):
        # load the data
        scene_name = scene_result['scene_name']

        for obj_result in tqdm(scene_result['results'], desc='obj'):
            ins_id = obj_result['obj_id']

            if cases_list is not None:
                if (scene_result['scene_name'], obj_result['obj_id'], obj_result['sub_id_list'][0]) not in cases_list:
                    continue

            # get gt data
            dataset_sub = ScanNetSubset(sequence_dir, scene_name, ins_id, mask_path_root=mask_path_root)
            pts1_sampled_world = dataset_sub.get_gt_sampled_points_in_world(n_sample_pts)

            # DEBUG SAVE GT MESH: open3d.mesh
            gt_mesh = dataset_sub.get_gt_mesh()
            gt_mesh.compute_vertex_normals()
            #  save to ply
            # gt_mesh.export(os.path.join(save_dir, 'plot', scene_name, 'ins-'+str(ins_id), 'gt_mesh.ply'))
            o3d.io.write_triangle_mesh(os.path.join(save_dir, 'plot', scene_name, 'ins-'+str(ins_id), 'gt_mesh.ply'), gt_mesh)

            if vis_mesh:
                save_im_name = os.path.join(save_dir, 'plot', scene_name, 'ins-'+str(ins_id), 'gt_mesh.png')
                visualize_o3d_ins_to_image([gt_mesh], './view_file_deepsdf.json', save_im_name, vis)
            

            for frame_result in obj_result['results']:
                n_all_data += 1

                frame_id = frame_result['sub_id']

                results = frame_result['results']
                obj_data = results[0]

                save_dir_plot = os.path.join(save_dir, 'plot', scene_name, 'ins-'+str(ins_id), 'frame-'+str(frame_id))
                os.makedirs(save_dir_plot, exist_ok=True)

                if not obj_data['is_good']:
                    continue

                # Test one object
                loss = evaluate_instance(obj_data, pts1_sampled_world, decoder, 
                                         save_dir=save_dir_plot, th=th, vis_mesh=vis_mesh, vis=vis)

                if loss is not None:

                    # save the loss to the result
                    evo_result_list.append({
                        'scene_name': scene_name,
                        'ins_id': ins_id,
                        'frame_id': frame_id,
                        'loss': loss
                    })
                
                else:

                    fail_cases.append(
                        {
                            'scene_name': scene_name,
                            'ins_id': ins_id,
                            'frame_id': frame_id,
                            'reason': f'no points inside th {th}'
                        }
                    )

    # statistics analysis of those results
    # consider those success ones (this needs the evo before? Has it been saved?)

    print('output summarize of the whole scene...')

    # Average the normal, es, nll of all objects
    normal_list = [data['loss']['normal'] for data in evo_result_list]
    es_list = [data['loss']['es'] for data in evo_result_list]
    nll_list = [data['loss']['nll'] for data in evo_result_list]

    pearson_list = [data['loss']['pearson'] for data in evo_result_list]
    pearson_bins_list = [data['loss']['pearson_bins'] for data in evo_result_list]

    aver_normal = np.mean(normal_list)
    aver_es = np.mean(es_list)
    aver_nll = np.mean(nll_list)

    aver_pearson = np.mean(pearson_list)
    aver_pearson_bins = np.mean(pearson_bins_list)

    std_normal = np.std(normal_list)
    std_es = np.std(es_list)
    std_nll = np.std(nll_list)

    std_pearson = np.std(pearson_list)
    std_pearson_bins = np.std(pearson_bins_list)

    print('normal aver/std: ', aver_normal, '/', std_normal)
    print('es aver/std: ', aver_es, '/', std_es)
    print('nll aver/std: ', aver_nll, '/', std_nll)
    print('pearson aver/std: ', aver_pearson, '/', std_pearson)
    print('pearson_bins aver/std: ', aver_pearson_bins, '/', std_pearson_bins)

    n_valid_data = len(evo_result_list)
    print('n_valid_data: ', n_valid_data, '/', n_all_data)

    print('All finished')

    output = {
        'evo_result_list': evo_result_list,
        'fail_cases': fail_cases,
        'aver_normal': aver_normal,
        'aver_es': aver_es,
        'aver_nll': aver_nll,
        'aver_pearson': aver_pearson,
        'aver_pearson_bins': aver_pearson_bins,
        'std_normal': std_normal,
        'std_es': std_es,
        'std_nll': std_nll,
        'std_pearson': std_pearson,
        'std_pearson_bins': std_pearson_bins,
    }

    # also output some metrics for those succeed ones:
    # Trans/Rot/Scale, IoU, CD

    return output

def load_scene_name_data(views=3, category='chair'):
    '''
    Starting from 0914, we use the sliding window version multi-view. And use 3 views to test the uncertainty.

    @ output: dir .../results
    '''
    root_sing_dir = '/msrresrchsa/workspace/shape_pose/sing'

    # all evaluation, loaded from a config
    from postprocess.data_analyzer import load_exp_configs

    exp_configs = load_exp_configs(exp = 'all')

    # get the view 3, and ours 
    method_list = ['Ours-det', 'Ours-cov', 'Ours-a5']

    chair_list = {}
    table_list = {}
    for method in method_list:
        chair_list[method] = exp_configs[0][views][method]
        table_list[method] = exp_configs[1][views][method]

        # add a result_dataset_save.pth in the end
        chair_list[method] = os.path.join(chair_list[method], 'result_dataset_save.pth')
        table_list[method] = os.path.join(table_list[method], 'result_dataset_save.pth')

    return {
        'chair': chair_list,
        'table': table_list
    }[category]


def print_metrics_to_table_with_expnames(success_list_exp_metrics):
    
    print('exp_name, trans, std, rot, std, scale, std, iou, std, cd, std, ob_ratio, std, num_valid')
    for exp_name, data in success_list_exp_metrics.items():
        # success_metrics.append([row[3],row[4],row[5],row[7],row[8],row[9]])
        data = np.asarray(data)
        
        # Add a row
        # exp_name, trans, std, rot, std, scale, std, iou, std, cd, std, ob_ratio, std, num_valid
        trans_mean = np.mean(data[:,0])
        trans_std = np.std(data[:,0])
        rot_mean = np.mean(data[:,1])
        rot_std = np.std(data[:,1])
        scale_mean = np.mean(data[:,2])
        scale_std = np.std(data[:,2])
        iou_mean = np.mean(data[:,3])
        iou_std = np.std(data[:,3])
        cd_mean = np.mean(data[:,4])
        cd_std = np.std(data[:,4])
        ob_ratio_mean = np.mean(data[:,5])
        ob_ratio_std = np.std(data[:,5])

        num_valid = len(data)

        # print to text, split by ','
        variable_values = [trans_mean, trans_std, rot_mean, rot_std, scale_mean, scale_std, iou_mean, iou_std, cd_mean, cd_std, ob_ratio_mean, ob_ratio_std, num_valid]

        # Create a string with the variable names and values
        variable_string = ', '.join([f'{value}' for value in variable_values])

        print(f'{exp_name}, {variable_string}')


def debug_scene_result(evo_uncertainty=True, evo_accuracy=False, vis_mesh=False, th = 0.2,
                           save_dir_root = 'output/debug/uncertain_3d_0912_all',
                        success_condition = 'success', views=1, category='chair'):
    '''

    @ success_condition: success / cd-0.1 / cd-0.2
    '''

    scene_name_data_list = load_scene_name_data(views, category)
    os.makedirs(save_dir_root, exist_ok=True)

    save_file_name = f'all_cond-{success_condition}_th-{th}.pt'

    ######

    output_save_dict = {
        'configs': {
            'th': th
        }
    }

    # Step 1: Get the success cases!
    success_list_exp = {}
    for exp_name, save_data_path in scene_name_data_list.items():
        print('exp_name: ', exp_name)

        # file name: result_dataset_save_w_evo.pth
        if 'result_dataset_save.pth' in save_data_path:
            save_data_with_evo = save_data_path.replace('result_dataset_save.pth', 'collect_data.csv')
        else:
            save_data_with_evo = os.path.join(save_data_path, 'collect_data.csv')
        import pandas as pd
        # no description row
        data_w_evo = pd.read_csv(save_data_with_evo, header=None)

        # find success cases: scene_name, obj_id, frame_id
        success_cases = []
        for i in range(len(data_w_evo)):
            row = data_w_evo.iloc[i]
            
            if success_condition == 'success':
                success = row[6] == True
            elif success_condition == 'cd-0.1':
                # success, and cd
                success = (row[8] < 0.1 and row[6] == True)
            elif success_condition == 'cd-0.2':
                success = (row[8] < 0.2 and row[6] == True)
            elif success_condition == 'cd-0.15':
                success = (row[8] < 0.15 and row[6] == True)
            elif success_condition == 'cd-0.08': # 0.05, none objects
                # success, and cd
                success = (row[8] < 0.08 and row[6] == True)
            if success:
                success_cases.append((row[0],row[1],row[2]))

        # add success cases to bigger list
        success_list_exp[exp_name] = success_cases
    
    # Process those common success ones
    # get the common success cases
    common_success_cases = None
    for exp_name, success_cases in success_list_exp.items():
        if common_success_cases is None:
            common_success_cases = success_cases
        else:
            common_success_cases = [case for case in success_cases if case in common_success_cases]
    
    print('common_success_cases: ', len(common_success_cases))

    # only get those common success cases
    success_list_exp_metrics = {}
    for exp_name, save_data_path in scene_name_data_list.items():
        print('exp_name: ', exp_name)

        # file name: result_dataset_save_w_evo.pth
        if 'result_dataset_save.pth' in save_data_path:
            save_data_with_evo = save_data_path.replace('result_dataset_save.pth', 'collect_data.csv')
        else:
            save_data_with_evo = os.path.join(save_data_path, 'collect_data.csv')
        import pandas as pd
        # no description row
        data_w_evo = pd.read_csv(save_data_with_evo, header=None)

        # find success cases: scene_name, obj_id, frame_id
        success_metrics = []
        for i in range(len(data_w_evo)):
            row = data_w_evo.iloc[i]
            if row[6] == True and (row[0],row[1],row[2]) in common_success_cases:
                # store trans,rot,scale,iou,cd,ob_ratio
                success_metrics.append([row[3],row[4],row[5],row[7],row[8],row[9]])
        # add success cases to bigger list
        success_list_exp_metrics[exp_name] = success_metrics

    # Analyze the Average and Std of Metrics of Every method into table
    print_metrics_to_table_with_expnames(success_list_exp_metrics)

    # dir of save_data_path
    exp_name_with_th = 'uncertain_3d_all'
    if th is not None:
        exp_name_with_th += '-th'+str(th)

    for exp_name, save_data_path in scene_name_data_list.items():
        print('exp_name: ', exp_name)

        data = torch.load(save_data_path)

        save_dir = os.path.join(os.path.dirname(save_data_path), exp_name_with_th)

        output = io_evaluate_scene_result(data, save_dir=save_dir, th=th, cases_list=common_success_cases, 
                                        vis_mesh=vis_mesh, category=category)

        output_save_dict[exp_name] = output

    # specify a scene; and only visualize for sdf under a thresh
    # save dict to local

    save_file_full_name = os.path.join(save_dir_root, save_file_name)
    torch.save(output_save_dict, save_file_full_name)

    print('all done')

    return save_file_full_name

def output_average_uncertainty_3d(method_data):
    fail_num = len(method_data['fail_cases'])
    success_num = len(method_data['evo_result_list'])

    print('fail_num: ', fail_num)
    print('success_num: ', success_num)

    # average normal/es/nll, std normal/es/nll
    print('aver_normal: ', method_data['aver_normal'], '/', method_data['std_normal'])
    print('aver_es: ', method_data['aver_es'], '/', method_data['std_es'])
    print('aver_nll: ', method_data['aver_nll'], '/', method_data['std_nll'])

    # go through all pearson, and calculate nanmean
    pearson_list = [data['loss']['pearson'] for data in method_data['evo_result_list']]
    pearson_aver = np.nanmean(pearson_list)
    pearson_std = np.nanstd(pearson_list)

    print('pearson_aver: ', pearson_aver, '/', pearson_std)
    
    # Only consider success cases, with trans/rot/scale < 0.2/20deg/0.2

def analyze_run_result():

    result_name = 'output/debug/obj.pt'

    result = torch.load(result_name)

    for method in result.keys():
        if method == 'configs':
            continue
        method_data = result[method]

        print('method: ', method)

        output_average_uncertainty_3d(method_data)


def print_result_to_table(save_file_full_name):
    data = torch.load(save_file_full_name)

    detail = True

    # print to table
    # method, normal, std, es, std, nll, std, pearson, std, pearson_bins, std, n_valid, n_all
    for exp_name, output in data.items():
        if exp_name == 'configs':
            continue

        # print to table
        print(exp_name, ', ', 
            f'{output["aver_normal"]}', ', ', f'{output["std_normal"]}', ', ', 
            f'{output["aver_es"]}', ', ', f'{output["std_es"]}', ', ', 
            f'{output["aver_nll"]}', ', ', f'{output["std_nll"]}', ', ', 
            f'{output["aver_pearson"]}', ', ', f'{output["std_pearson"]}', ', ', 
            f'{output["aver_pearson_bins"]}', ', ', f'{output["std_pearson_bins"]}', ', ', 
            f'{len(output["evo_result_list"])}', ', ', f'{len(output["fail_cases"])}')

        # if print detail
        if detail:
            # print into an indenpend text file for every line
            file_name = f'output/{exp_name}.txt'

            data_detail = output['evo_result_list']
            with open(file_name, 'w') as f:
                f.write('scene_name, ins_id, frame_id, normal, es, nll, pearson, pearson_bins\n')
                for data in data_detail:
                    f.write(f'{data["scene_name"]}, {data["ins_id"]}, {data["frame_id"]}, {data["loss"]["normal"]}, {data["loss"]["es"]}, {data["loss"]["nll"]}, {data["loss"]["pearson"]}, {data["loss"]["pearson_bins"]}\n')

            

# parse arguments
def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--th', type=float, default=None, help='threshold for sdf')
    parser.add_argument('--success_condition', type=str, default='success', help='success / cd-0.1 / cd-0.2')

    # add task
    parser.add_argument('--task', default='run')

    parser.add_argument('--save_file', default=None)

    parser.add_argument('--views', type=int, default=1)

    parser.add_argument('--category', type=str, default='chair')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = arg_parser()
    # debug()

    root_dir = 'output/debug'

    # debug, only this task
    args.task = 'one_instance'

    if args.task == 'run':

        th = args.th
        success_condition = args.success_condition
        views = args.views

        category = args.category

        print('th: ', th)
        print('success_condition: ', success_condition)

        save_dir_root = os.path.join(root_dir, f'uncertain_3d_all_v{views}_close_{category}')

        save_file_full_name = debug_scene_result(evo_uncertainty=True, evo_accuracy=True, vis_mesh=False, save_dir_root=save_dir_root, 
                        th = th, success_condition=success_condition, views=views, category=category)

        print_result_to_table(save_file_full_name)
    
    elif args.task == 'print':
        '''
        all_cond-{success_condition}_th-{th}
        '''
        print_result_to_table(args.save_file)

    elif args.task == 'one_instance':
        '''
        Reconstruct the error mesh, uncertainty mesh, and the sdf-error plot, and save the sdf-error origin data.
        with a given method.
        '''

        # the method: we use Ours-a5, 3 views.
        result_save_dir = 'results/result_dataset_save.pth'
        scene_recon_result = torch.load(result_save_dir)

        th = 0.15

        common_success_cases = [
            ('scene0633_01', 4, 7),
        ]

        vis_mesh = True
        category = 'chair'

        save_dir = './output/obj_vis_uncertain'

        output = io_evaluate_scene_result(scene_recon_result, save_dir=save_dir, th=th, cases_list=common_success_cases, 
                                vis_mesh=vis_mesh, category=category)

    print('DONE.')


