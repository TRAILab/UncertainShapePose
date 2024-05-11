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

import math
import numpy as np
import torch
from utils import SE3
from reconstruct.utils import ForceKeyErrorDict, create_voxel_grid, convert_sdf_voxels_to_mesh
from reconstruct.loss import compute_sdf_loss, compute_render_loss, compute_rotation_loss_sim3, compute_sdf_loss_shape_pose, Oplus, \
                            compute_sdf_loss_uncertain, compute_render_loss_uncertain, compute_render_loss_nojac,\
                            compute_render_loss_uncertain_beta, compute_render_loss_uncertain_beta_jac, compute_render_loss_uncertain_jac_multiview, \
                            compute_render_loss_nojac_multiview
from reconstruct.loss_utils import decode_sdf, get_robust_res, exp_se3, exp_sim3, get_time

from torch.autograd import Variable,grad

from torch import distributions

from utils import SE3

def get_T_oc(T_oc_0, dPose, dScale):
    T_oc_1 = Oplus(T_oc_0, dPose)
    scale_4 = torch.cat([torch.exp(dScale), torch.tensor([1.0]).cuda()], -1)
    scaleMtx = torch.diag(scale_4)
    T_oc = torch.mm(T_oc_1, scaleMtx)
    T_oc_inter = T_oc.detach().cpu()
    return T_oc_inter

def random_select_data(pts_surface, iter_sample_num):
    '''
    @pts_surface: (N, 4)  x,y,z,sdf
    @iter_sample_num: n, 0<n<=N
    '''
    if iter_sample_num is None or iter_sample_num <= 0 or iter_sample_num > len(pts_surface):
        return pts_surface
    # check pos and neg
    pos_ind = torch.where(pts_surface[:,-1] >= 0)[0]
    neg_ind = torch.where(pts_surface[:,-1] < 0)[0]
    
    half = int(iter_sample_num / 2)
    pos_size = len(pos_ind)
    neg_size = len(neg_ind)
    
    all_size = pts_surface.shape[0]
    if pos_size >= half and neg_size >= half:
        # we assume pos and neg are larger than half
        pos_ind_select = torch.randperm(pos_size)[:half]
        neg_ind_select = torch.randperm(neg_size)[:half]

        pos_data = pts_surface[pos_ind, :]
        neg_data = pts_surface[neg_ind, :]

        pos_data_select = pos_data[pos_ind_select, :]
        neg_data_select = neg_data[neg_ind_select, :]

        valid_data = torch.cat([pos_data_select, neg_data_select], 0)
    else:
        # random select from all
        all_ind = torch.randperm(all_size)[:iter_sample_num]
        valid_data = pts_surface[all_ind, :]
        
    return valid_data

'''
Utils function
'''
def get_T_oc(T_ow_0, dPose, dScale, T_wc_vis):
    T_ow_1 = Oplus(T_ow_0, dPose)
    scale_4 = torch.cat([torch.exp(dScale), torch.tensor([1.0]).cuda()], -1)
    scaleMtx = torch.diag(scale_4)
    T_ow = torch.mm(T_ow_1, scaleMtx)
    T_oc = torch.mm(T_ow, T_wc_vis)
    T_oc_inter = T_oc.detach().cpu()
    return T_oc_inter

class Optimizer(object):
    def __init__(self, decoder, configs):
        self.decoder = decoder # Checked it's .eval() in workspace.py
        optim_cfg = configs.optimizer
        self.k1 = optim_cfg.joint_optim.k1
        self.k2 = optim_cfg.joint_optim.k2
        self.k3 = optim_cfg.joint_optim.k3
        self.k4 = optim_cfg.joint_optim.k4
        self.b1 = optim_cfg.joint_optim.b1
        self.b2 = optim_cfg.joint_optim.b2
        self.k_norm = optim_cfg.joint_optim.k_norm
        self.lr = optim_cfg.joint_optim.learning_rate
        self.s_damp = optim_cfg.joint_optim.scale_damping
        self.num_iterations_joint_optim = optim_cfg.joint_optim.num_iterations
        self.code_len = optim_cfg.code_len
        self.num_depth_samples = optim_cfg.num_depth_samples
        self.cut_off = optim_cfg.cut_off_threshold
        if configs.data_type == "KITTI":
            self.num_iterations_pose_only = optim_cfg.pose_only_optim.num_iterations

    def estimate_pose_cam_obj(self, t_co_se3, scale, pts, code):
        """
        :param t_co_se3: o2c transformation (4, 4) in SE(3)
        :param scale: object scale
        :param pts: surface points (M, 3)
        :param code: shape code
        :return: optimized o2c transformation
        """
        t_cam_obj = torch.from_numpy(t_co_se3)
        t_cam_obj[:3, :3] *= scale
        t_obj_cam = torch.inverse(t_cam_obj)
        latent_vector = torch.from_numpy(code).cuda()
        pts_surface = torch.from_numpy(pts).cuda()

        for e in range(self.num_iterations_pose_only):
            start = get_time()
            # 1. Compute SDF (3D) loss
            de_dsim3_sdf, de_dc_sdf, res_sdf = \
                compute_sdf_loss(self.decoder, pts_surface,
                                      t_obj_cam,
                                      latent_vector)
            _, sdf_loss, _ = get_robust_res(res_sdf, 0.05)

            j_sdf = de_dsim3_sdf[..., :6]
            hess = torch.bmm(j_sdf.transpose(-2, -1), j_sdf).sum(0).squeeze().cpu() / j_sdf.shape[0]
            hess += 1e-2 * torch.eye(6)
            b = -torch.bmm(j_sdf.transpose(-2, -1), res_sdf).sum(0).squeeze().cpu() / j_sdf.shape[0]
            dx = torch.mv(torch.inverse(hess), b)
            delta_t = exp_se3(dx)
            t_obj_cam = torch.mm(delta_t, t_obj_cam)

            if e == 4:
                inliers_mask = torch.abs(res_sdf).squeeze() <= 0.05
                pts_surface = pts_surface[inliers_mask, :]

            # print("Object pose-only optimization: Iter %d, sdf loss: %f" % (e, sdf_loss))

        # Convert back to SE3
        t_cam_obj = torch.inverse(t_obj_cam)
        t_cam_obj[:3, :3] /= scale

        return t_cam_obj

    def reconstruct_shape_pose_world(self, t_world_obj_0, detections, code=None, sample_num=10,
        loss_type="NLL", num_iterations=10, lr=1.0,
        save_intermediate=False,
        device = 'cuda',
        init_sigma = 0.01,
        use_2d_loss = False,
        loss_type_2d_uncertain = 'energy_score',
        lr_scheduler=None,
        init_sigma_pose = 0.01,
        init_sigma_scale = 0.01,
        render_2d_K=400,
        render_2d_calibrate_C=1.0,
        render_2d_const_a=None,
        b_optimize_pose=True):
        '''
        @ t_world_obj_0: initialized pose of the object in world frame; 4x4 matrix in SE(3) + Scale (3)

        @ init_sigma: CODE sigma; Note that pose sigma is always 0.01 now.
        @ init_sigma_pose: sigma for pose 6x1, + 3x1 scale
        '''

        # Code Initialization
        ########################################################################################################
        if code is None:
            init_sigma_log = math.log(init_sigma * init_sigma)
            # dsp-slam method
            latent_vector_distribution_init = torch.zeros(self.code_len, 2, device=device)
            # set sigma to 1; uncertainty is stored as log(var)
            latent_vector_distribution_init[..., 1] = init_sigma_log
            latent_vector_distribution = latent_vector_distribution_init.clone().detach().requires_grad_(True)
        else:
            latent_vector_distribution = code.clone().detach().requires_grad_(True)

        code_square_mean = latent_vector_distribution[:, 0].square().mean().detach().item()
        uncer_square_mean = latent_vector_distribution[:, 1].exp().mean().detach().item()
        print('[Initial Code]', 'code_square_mean:', round(code_square_mean, 4), 'uncer_square_mean:', round(uncer_square_mean, 4))
        ########################################################################################################

        # Init pose of the object in world frame
        N_obs = len(detections)
        # print("Number of observations", N_obs)

        # Initialize transform of object in world frame
        ##########################################################
        # Decouple the scale from input pose; Note: scale is in object frame
        t_obj_world_0 = np.linalg.inv(t_world_obj_0)
        t_ow, q_ow, s_ow = SE3.decompose_mat4(t_obj_world_0) 
        t_obj_world = SE3.compose_mat4(t_ow, q_ow, np.ones(3))

        T_ow_0 = torch.from_numpy(t_obj_world).float().cuda()   # SE(3) w/o scale

        ref_frame_id = 0   # take the first frame as ref frame, e.g., use it's camera pose in world to visualize and store

        # use last/current camera pose for visualization
        T_wc_vis = torch.from_numpy(detections[ref_frame_id].T_world_cam).float().cuda()
        T_cw_vis = torch.inverse(T_wc_vis)
        ##########################################################

        # Pose Initialization
        ##################################################################################
        ## 6-DOF Pose
        sigma_init = init_sigma_pose  # pose sigma is fixed and not changed yet
        sigma_log = math.log(sigma_init * sigma_init)
        pose_ini = torch.tensor([0, 0, 0, 1e-12, 1e-12, 1e-12],
                                 dtype=torch.float32, device=device)
        pose_std_ini = torch.tensor([sigma_log, sigma_log, sigma_log,
                                     sigma_log, sigma_log, sigma_log],
                                 dtype=torch.float32, device=device)
        dPose = pose_ini.clone().detach().requires_grad_(b_optimize_pose)
        dPose_std = pose_std_ini.clone().detach().requires_grad_(b_optimize_pose)

        ## 3-DOF Scale
        scale_log_x = math.log(s_ow[0])
        scale_log_y = math.log(s_ow[1])
        scale_log_z = math.log(s_ow[2])

        sigma_scale = init_sigma_scale
        sigma_scale_log = math.log(sigma_scale * sigma_scale)
        scale_ini = torch.tensor([scale_log_x, scale_log_y, scale_log_z],
                                 dtype=torch.float32, device=device)   # log
        scale_std_ini = torch.tensor([sigma_scale_log, sigma_scale_log, sigma_scale_log],
                                 dtype=torch.float32, device=device)   # log(var)
        dScale = scale_ini.clone().detach().requires_grad_(b_optimize_pose)
        dScale_std = scale_std_ini.clone().detach().requires_grad_(b_optimize_pose)
        ##################################################################################

        # Optimizer
        #########################################################################
        optimizer = torch.optim.Adam([latent_vector_distribution, dPose, dScale,
                                      dPose_std, dScale_std], lr=lr)
        if lr_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)   # TODO: Ablade the effectiveness
        else:
            scheduler = None
        #########################################################################

        # surface points within Omega_s (Object Mask)
        ##############################################################################
        # sample points from all available observations
        pts_surface_world = np.vstack( [det.surface_points_world for det in detections] )
        pts_surface_world = np.hstack( (pts_surface_world, np.ones((pts_surface_world.shape[0], 1))) )
        pts_surface = torch.from_numpy(pts_surface_world).float().cuda()  # in world 

        ##############################################################################
        # Outputs
        ######################################################################
        loss = torch.Tensor([0.])
        intermedia_values = []
        intermedia_values.append(latent_vector_distribution.detach().cpu())
        intermedia_values_pose = []
        
        # construct a pose with scale: 4x4
        T_oc_inter = get_T_oc(T_ow_0, dPose, dScale, T_wc_vis)
        intermedia_values_pose.append(T_oc_inter)

        # also add pose uncertainty  3+3+3 x 1
        intermedia_values_pose_std = []
        intermedia_values_pose_std.append(pose_std_ini.detach().cpu())

        # add scale
        intermedia_values_scale = []
        intermedia_values_scale.append(dScale.detach().cpu())

        # scale uncertainty
        intermedia_values_scale_std = []
        intermedia_values_scale_std.append(scale_std_ini.detach().cpu())

        loss_list = []
        normal_loss_list = []  # for uncertainty loss, save normal loss
        sigma_mean_list = []  # store uncertainty scale
        sdf_loss_list = []
        norm_term_list = []
        loss_2d_list = []
        ######################################################################

        # Preprocess for using 2d loss
        use_2d_loss = True
        ray_directions = []
        n_foreground_rays = []
        n_background_rays = []
        depth_obs = []
        if use_2d_loss:
            for det in detections:
                ray_directions.append(torch.from_numpy(det.rays).cuda())
                n_foreground_rays.append(det.depth.shape[0])
                n_background_rays.append(det.rays.shape[0] - det.depth.shape[0])
                depth_obs.append(torch.from_numpy(np.concatenate([det.depth, np.zeros(n_background_rays[-1])], axis=0).astype(np.float32)).cuda())

        pts_local = None
        #**************************************************************************************************************
        for e in range(num_iterations):
            optimizer.zero_grad()

            # Camera Pose
            #########################################################################
            T_ow_1 = Oplus(T_ow_0, dPose)  # w/o scale
            scale_4 = torch.cat([torch.exp(dScale), torch.tensor([1.0]).cuda()], -1)
            scaleMtx = torch.diag(scale_4)
            T_ow = torch.mm(T_ow_1, scaleMtx) # w/ scale   # The definition of Scale depends on how we use it to get T_ow. Right-Mult makes it scale w.r.t. the zero coordinate of world in object frame. When transforming points in world to object, first it will scale in world frame then transform.
            #########################################################################


            # 1. Compute SDF (3D) loss
            #######################################################################################
            sdf_rst = compute_sdf_loss_shape_pose(self.decoder, pts_surface, latent_vector_distribution, T_ow, T_ow_1,
                                                  dPose_std, dScale_std, loss_type=loss_type, T_vis=T_cw_vis)

            res_sdf = sdf_rst.loss  # we only output one loss
            pts_local = sdf_rst.pts_local
            normal_loss = sdf_rst.normal_loss
            sdf_loss = res_sdf

            #######################################################################################

            use_norm = True
            if use_norm:
                norm_term_coeff = 1.0
                latent = latent_vector_distribution[:, 0]
                norm_term = norm_term_coeff * torch.mean(latent.pow(2))
            else:
                norm_term = 0  # no norm

            # 3. 2d render loss
            ####################################################################################################
            loss_2d = torch.tensor(0.0).cuda()
            
            if use_2d_loss:
                T_wo = torch.inverse(T_ow)      # w/ scale # T_ow = T_ow_1 @ scaleMtx ; T_wo = scaleMtx_inv @ T_ow_1_inv
                T_wo_1 = torch.inverse(T_ow_1)  # w/o scale

                ray_directions_mv = [] 
                depth_obs_mv = []
                T_oc_mv = []
                T_oc_1_mv = []
                sampled_depth_along_rays_mv = []
                background_depth_mv = []

                for i, det in enumerate(detections):
                    # get depth range and sample points along the rays
                    T_wc_i = torch.from_numpy(det.T_world_cam).cuda()
                    T_cw_i = torch.inverse(T_wc_i)
                    T_co_i = torch.mm(T_cw_i, T_wo)
                    T_co_1_i = torch.mm(T_cw_i, T_wo_1)
                    scale = torch.det(T_co_i[:3, :3]) ** (1 / 3)
                    depth_min, depth_max = T_co_i[2, 3] - 1.0 * scale, T_co_i[2, 3] + 1.0 * scale
                    sampled_depth_along_rays = torch.linspace(depth_min.item(), depth_max.item(), self.num_depth_samples).cuda()
                    # set background depth to d'

                    background_depth_i = (1.1 * depth_max.detach().cpu().item())
                    background_depth_mv.append(background_depth_i)
                    
                    '''
                    depth_max accidently introduces gradients tracking, because of T_co, and scale.
                    We shoule make it a constant variable manually.
                    '''
                    # BACKGROUND_DEPTH = BACKGROUND_DEPTH.item()
                    depth_obs[i][n_foreground_rays[i]:] = background_depth_i

                    num_total_rays = 1000  # originly we have 100k; limit it to 1/100
                    ratio_foreground = 0.5  # 500 foreground and 500 background
                    num_foreground_rays_iter = round(num_total_rays * ratio_foreground / N_obs)
                    num_background_rays_iter = round(num_total_rays / N_obs) - num_foreground_rays_iter

                    # sample foreground and background points
                    if num_foreground_rays_iter < n_foreground_rays[i]:
                        random_foreground_idx = np.random.choice(n_foreground_rays[i], num_foreground_rays_iter, replace=False)
                    else:
                        # select all
                        random_foreground_idx = np.arange(n_foreground_rays[i])

                    if num_background_rays_iter < n_background_rays[i]:
                        random_background_idx = np.random.choice(n_background_rays[i], num_background_rays_iter, replace=False)
                    else:
                        # select all
                        random_background_idx = np.arange(n_background_rays[i])
                    random_idx = np.concatenate([random_foreground_idx, random_background_idx + n_foreground_rays[i]], axis=0)

                    # each iterations, we sample different points
                    ray_directions_iter = ray_directions[i][random_idx]
                    depth_obs_iter = depth_obs[i][random_idx]

                    ray_directions_mv.append(ray_directions_iter) 
                    depth_obs_mv.append(depth_obs_iter)
                    T_oc_mv.append(torch.inverse(T_co_i))
                    T_oc_1_mv.append(torch.inverse(T_co_1_i))
                    sampled_depth_along_rays_mv.append(sampled_depth_along_rays)

                if loss_type_2d_uncertain == 'dsp' or loss_type_2d_uncertain == 'node' or loss_type_2d_uncertain == 'node_es':
                    # DSP-SLAM
                    ##################################################################################################
                    latent_vector = latent_vector_distribution[:, 0]
                    render_loss = compute_render_loss_nojac_multiview(self.decoder, ray_directions_mv, depth_obs_mv, T_oc_mv, # TODO: check T_oc_mv input
                                    sampled_depth_along_rays_mv, latent_vector, th=self.cut_off, loss_type_2d_uncertain=loss_type_2d_uncertain)

                    ##################################################################################################
                else:
                    render_rst = compute_render_loss_uncertain_jac_multiview(self.decoder, ray_directions_mv, depth_obs_mv,
                                                                    T_oc_mv, T_oc_1_mv, dPose_std, dScale_std,
                                                                    sampled_depth_along_rays_mv, latent_vector_distribution,
                                                                    th=self.cut_off, sample_num=sample_num,
                                                                    loss_type=loss_type_2d_uncertain,
                                                                    dtype=torch.float64,
                                                                    background_depth_mv=np.array(background_depth_mv),
                                                                    render_2d_K=render_2d_K,
                                                                    render_2d_calibrate_C=render_2d_calibrate_C,
                                                                    render_2d_const_a=render_2d_const_a)
                    
                    render_loss = render_rst
                    #####################################################################################################
                loss_2d = render_loss
                # TODO: Check if this is correct
                if loss_2d is None or math.isnan(loss_2d):
                    print('2d loss is None or nan')
                    return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)

            # Check if it's from 2d or 3d loss
            loss = self.k2 * sdf_loss + self.k_norm * norm_term + self.k1 * loss_2d
            
            loss_list.append(loss.detach().item())
            normal_loss_list.append(normal_loss.detach().item())
            sigma_mean_list.append(latent_vector_distribution[:, 1].detach().exp().sqrt().mean().item())
            sdf_loss_list.append(sdf_loss.detach().item())
            if use_norm:
                norm_term_list.append(norm_term.detach().item())
            if use_2d_loss:
                loss_2d_list.append(loss_2d.detach().item())
            #############################################################################################

            loss.backward()  # show error

            clip_val = 0.1
            torch.nn.utils.clip_grad_norm_(latent_vector_distribution, clip_val)
            torch.nn.utils.clip_grad_norm_(dPose, clip_val)
            torch.nn.utils.clip_grad_norm_(dScale, clip_val)
            torch.nn.utils.clip_grad_norm_(dPose_std, clip_val)
            torch.nn.utils.clip_grad_norm_(dScale_std, clip_val)

            optimizer.step()
            if lr_scheduler is not None:
                scheduler.step()

            N_OUTPUT_TIMES = 5
            if e % (num_iterations / N_OUTPUT_TIMES) == 0 or e == num_iterations - 1:
                grad = optimizer.param_groups[0]['params'][0].grad
                code_square_mean = latent_vector_distribution[:, 0].square().mean().detach().item()
                uncer_square_mean = latent_vector_distribution[:, 1].exp().sqrt().square().mean().detach().item()
                print(f'[it {e}] {loss_type} loss:', "{:.2f}".format(loss.item()),
                    'sdf loss:', "{:.3f}".format(sdf_loss.item()),
                    'loss_2d:', "{:.3f}".format(loss_2d.item()),
                    'norm_term:', "{:.4f}".format(norm_term.item()),
                    'sdf normal:', "{:.4f}".format(normal_loss.item()),
                    'gd-norm:', "{:.2f}".format(grad.norm().item()),
                    'code_square_mean:', "{:.4f}".format(code_square_mean),
                    'uncer_square_mean:', "{:.2g}".format(uncer_square_mean))

            if save_intermediate:
                dist_sigma_inter = latent_vector_distribution.detach().cpu()
                intermedia_values.append(dist_sigma_inter)

                T_oc_inter = get_T_oc(T_ow_0, dPose, dScale, T_wc_vis)
                intermedia_values_pose.append(T_oc_inter)

                pose_std_inter = dPose_std.detach().cpu()
                intermedia_values_pose_std.append(pose_std_inter)

                scale_std_inter = dScale_std.detach().cpu()
                intermedia_values_scale_std.append(scale_std_inter)


        '''
        Iteration Done
        '''
        t_obj_world = Oplus(T_ow_0, dPose)
        scale_4 = torch.cat([torch.exp(dScale), torch.tensor([1.0]).cuda()], -1)
        scaleMtx = torch.diag(scale_4)
        t_obj_world = torch.mm(t_obj_world, scaleMtx)

        t_world_obj = torch.inverse(t_obj_world)
        latent_code = latent_vector_distribution.detach().clone()
        dist_sigma = latent_vector_distribution.detach().clone()
        # change log(var) to sigma
        dist_sigma[:,1] = dist_sigma[:,1].exp().sqrt()
        dist_sigma = dist_sigma.cpu().numpy()

        t_cam_obj = torch.mm(T_cw_vis, t_world_obj) # this is similar to T_cam_deepsdf

        # change log(var) to sigma
        for i in range(len(intermedia_values)):
            intermedia_values[i][:,1] = intermedia_values[i][:,1].exp().sqrt()
        intermedia_output_shape_pose = {
            'code': intermedia_values,
            'T_oc': intermedia_values_pose,
            'T_oc_std': intermedia_values_pose_std,
            'scale_std': intermedia_values_scale_std
        }
        return ForceKeyErrorDict(t_world_obj=t_world_obj.detach().cpu().numpy(),
                                 t_cam_obj=t_cam_obj.detach().cpu().numpy(),
                                 latent_code=latent_code, code=dist_sigma,
                                 is_good=True, loss=loss.detach().item(), intermediate=intermedia_output_shape_pose, pts_local=pts_local,
                                 loss_list=loss_list, normal_loss_list=normal_loss_list, sigma_mean_list=sigma_mean_list,
                                 loss_sdf_list=sdf_loss_list, loss_norm_list=norm_term_list, loss_2d_list=loss_2d_list)


class MeshExtractor(object):
    def __init__(self, decoder, code_len=64, voxels_dim=64):
        self.decoder = decoder
        self.code_len = code_len
        self.voxels_dim = voxels_dim
        with torch.no_grad():
            self.voxel_points = create_voxel_grid(vol_dim=self.voxels_dim).cuda()

    def extract_mesh_from_code(self, code):
        '''
        code: support numpy or torch tensor
        '''
        start = get_time()
        latent_vector = code[:self.code_len]
        if isinstance(latent_vector, np.ndarray):
            latent_vector = torch.from_numpy(latent_vector).cuda()
        latent_vector = latent_vector.cuda()
        sdf_tensor = decode_sdf(self.decoder, latent_vector, self.voxel_points)
        vertices, faces = convert_sdf_voxels_to_mesh(sdf_tensor.view(self.voxels_dim, self.voxels_dim, self.voxels_dim))
        vertices = vertices.astype("float32")
        faces = faces.astype("int32")
        end = get_time()
        # print("Extract mesh takes %f seconds" % (end - start))
        return ForceKeyErrorDict(vertices=vertices, faces=faces)

    '''
    utils codes from Uncertainty-Mesh Extractor
    '''
    def decode_code_list_batch(self, code_with_sample, sample_points=None, max_batch=64**3):
        '''
        Args:
            code_with_sample: Batch, N, dims
            max_batch: Please check the maximum value according to your GPUs.
            
            sample_points: batch, n_pts, 3
            
        Output:
            sdf_tensors: batch, n_code_sample, num_pts, 1
        '''
        # N = len(code_list)
        # sdf_tensor_list = []

        if sample_points is None:
            sample_points = self.voxel_points
            # sample_points: batch, n_pts, 3
            sample_points = sample_points.unsqueeze(0).expand(code_with_sample.shape[0],-1,-1)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        batch,n_code_sample,dims = code_with_sample.shape
        batch,num_pts,_ = sample_points.shape
        
        # input; give each points a code
        code_with_sample_expand = code_with_sample.unsqueeze(2).expand(-1,-1,num_pts,-1) # Batch, code_sample, dims -> Batch, code_sample, num_pts, 3
        lat_vec_batch = code_with_sample_expand.flatten(0,2)  # Batch, code_sample, dims -> Batch * num_pts, dims
        
        # input
        sample_points_expand = sample_points.unsqueeze(1).expand(-1, n_code_sample, -1, -1)  # Batch, num_pts, 3  ->  Batch, code_sample, num_pts, 3
        sample_points_flat = sample_points_expand.flatten(0,2)  # Batch, code_sample, num_pts, 3 -> Batch * code_sample * num_pts, 3

        sdf_tensor = self.decode_sdf_with_code_list(self.decoder, lat_vec_batch, x=sample_points_flat, max_batch=max_batch) # define your own voxel_points
        
        # pack back to sdf_tensor_list
        sdf_tensor_list = sdf_tensor.view(batch, n_code_sample, num_pts, 1)
        
        return sdf_tensor_list

    def decode_sdf_with_code_list(self, decoder, lat_vec_batch, x, max_batch=64**3):
        """
        Update: support different codes
        
        :param decoder: DeepSDF Decoder
        :param lat_vec: torch.Tensor (code_len,), latent code
        :param x: torch.Tensor (N, 3), query positions
        :return: batched outputs (N, )
        :param max_batch: max batch size
        :return:
        """

        num_samples = x.shape[0]

        head = 0

        # get sdf values given query points
        sdf_values_chunks = []
        with torch.no_grad():
            while head < num_samples:
                x_subset = x[head : min(head + max_batch, num_samples), 0:3].cuda()
                latent_subset = lat_vec_batch[head : min(head + max_batch, num_samples), :].cuda()

                fp_inputs = torch.cat([latent_subset, x_subset], dim=-1)
                sdf_values = decoder(fp_inputs).squeeze()

                sdf_values_chunks.append(sdf_values)
                head += max_batch

        sdf_values = torch.cat(sdf_values_chunks, 0).cuda()
        return sdf_values
    
    def sample_codes_and_decode_sdf_batch(self, code, code_sigma = 0.1, N = 10, sample_points=None, max_batch=64**3):       
        '''
        @ code: (batch,code_dim)
        @ sigma: (batch,code_dim) / None

        batch version. should be faster.
        @sample_points: if given, then calculate sdfs for those points; else, averagely sample in 3d grids.
        '''
        
        # batch,N,dims
        code_with_sample = self.sample_codes_batch(code,code_sigma,N)

        # pass through decoder to get sdfs
        sdf_list = self.decode_code_list_batch(code_with_sample, sample_points=sample_points, max_batch=max_batch)

        # get mean, sigma for each 3d points
        sdf_mean,sdf_sigma = self.calculate_mean_sigma_batch(sdf_list)

        return sdf_mean, sdf_sigma

    # sigma: constant for all; or a vector to specify each dimensions
    def sample_codes_batch(self,code,sigma,N):
        '''
        Update 2-12:
        Consider batch.
        
        @ code: (batch,code_dim)
        @ sigma: (batch,code_dim) / None
        
        return
        @ code with batch: (batch, N, code_dim)
        '''
        if sigma is None:
            return code.unsqueeze(1)

        cov_mat = torch.diag_embed(sigma.square())  # support torch > 1.13; now 1.9

        # use torch's reparameterization trick version random generation!
        # Define per-anchor Distributions
        multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
            code, cov_mat)
        # Define Monte-Carlo Samples  (N, batch, dims)
        distributions_samples = multivariate_normal_dists.rsample(
            (N,))

        # (batch, N, dims)
        distributions_samples = distributions_samples.transpose(0,1)
               
        return distributions_samples

    def calculate_mean_sigma_batch(self,sdf_list):
        '''
        :input sdf_list: batch, n_code_sample, num_pts, 1
        
        :output mean: batch, num_pts, 1
        :output sigma: batch, num_pts, 1
        '''
        if sdf_list is None:
            return None, None
        else:
            n_code_sample = sdf_list.shape[1]
            if n_code_sample == 1:
                # no sample et al. no uncertainty
                mean = sdf_list.squeeze(1)
                sigma = None
            else:
                mean = sdf_list.mean(axis=1)  # (Batch, samples, pts_num, 1) [32, 10, 49152, 1]
                # sigma = sdf_list.mean(axis=1) ###!
                sigma = sdf_list.std(axis=1)  
 
            return mean, sigma