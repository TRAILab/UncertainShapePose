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
from reconstruct.utils import ForceKeyErrorDict, convert_sdf_voxels_to_mesh
from reconstruct.loss import compute_sdf_loss, compute_render_loss, compute_rotation_loss_sim3
from reconstruct.loss_utils import decode_sdf, get_robust_res, exp_se3, exp_sim3, get_time

# for color
from matplotlib import cm

import open3d as o3d

from uncertainty.utils import prob_to_coeff

from reconstruct.optimizer import MeshExtractor

class Optimizer(object):
    def __init__(self, decoder, configs):
        self.decoder = decoder
        optim_cfg = configs.optimizer
        self.k1 = optim_cfg.joint_optim.k1
        self.k2 = optim_cfg.joint_optim.k2
        self.k3 = optim_cfg.joint_optim.k3
        self.k4 = optim_cfg.joint_optim.k4
        self.b1 = optim_cfg.joint_optim.b1
        self.b2 = optim_cfg.joint_optim.b2
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

    def reconstruct_object(self, t_cam_obj, pts, rays, depth, code=None):
        """
        :param t_cam_obj: object pose, object-to-camera transformation
        :param pts: surface points, under camera coordinate (M, 3)
        :param rays: sampled ray directions (N, 3)
        :param depth: depth values (K,) only contain foreground pixels, K = M for KITTI
        :return: optimized opject pose and shape, saved as a dict
        """
        # Always start from zero code
        if code is None:
            latent_vector = torch.zeros(self.code_len).cuda()
        else:
            latent_vector = torch.from_numpy(code[:self.code_len]).cuda()

        # Initial Pose Estimate
        t_cam_obj = torch.from_numpy(t_cam_obj)
        t_obj_cam = torch.inverse(t_cam_obj)
        # ray directions within Omega_r
        ray_directions = torch.from_numpy(rays).cuda()
        # depth observations within Omega_r
        n_foreground_rays = depth.shape[0]
        n_background_rays = rays.shape[0] - n_foreground_rays
        # print("rays: %d, total rays: %d" % (n_foreground_rays, n_background_rays))

        # attention: np.zeros will be replaced further in the for loop
        depth_obs = np.concatenate([depth, np.zeros(n_background_rays)], axis=0).astype(np.float32)
        depth_obs = torch.from_numpy(depth_obs).cuda()
        # surface points within Omega_s
        pts_surface = torch.from_numpy(pts).cuda()

        start = get_time()
        loss = 0.
        for e in range(self.num_iterations_joint_optim):
            # get depth range and sample points along the rays
            t_cam_obj = torch.inverse(t_obj_cam)
            scale = torch.det(t_cam_obj[:3, :3]) ** (1 / 3)
            # print("Scale: %f" % scale)
            depth_min, depth_max = t_cam_obj[2, 3] - 1.0 * scale, t_cam_obj[2, 3] + 1.0 * scale
            sampled_depth_along_rays = torch.linspace(depth_min, depth_max, self.num_depth_samples).cuda()
            # set background depth to d'
            depth_obs[n_foreground_rays:] = 1.1 * depth_max

            # 1. Compute SDF (3D) loss
            sdf_rst = compute_sdf_loss(self.decoder, pts_surface, t_obj_cam, latent_vector)
            if sdf_rst is None:
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)
            else:
                de_dsim3_sdf, de_dc_sdf, res_sdf = sdf_rst
            robust_res_sdf, sdf_loss, _ = get_robust_res(res_sdf, self.b2)
            if math.isnan(sdf_loss):
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)

            # 2. Compute Render (2D) Loss
            render_rst = compute_render_loss(self.decoder, ray_directions, depth_obs, t_obj_cam,
                                             sampled_depth_along_rays, latent_vector, th=self.cut_off)
            # in case rendering fails
            if render_rst is None:
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)
            else:
                de_dsim3_render, de_dc_render, res_render = render_rst

            # print("rays gradients on python side: %d" % de_dsim3_render.shape[0])
            robust_res_render, render_loss, _ = get_robust_res(res_render, self.b1)
            if math.isnan(render_loss):
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)

            # 3. Rotation prior
            drot_dsim3, res_rot = compute_rotation_loss_sim3(t_obj_cam)

            loss = self.k1 * render_loss + self.k2 * sdf_loss
            z = latent_vector.cpu()

            # Compute Jacobian and Hessia
            pose_dim = 7

            J_sdf = torch.cat([de_dsim3_sdf, de_dc_sdf], dim=-1)
            H_sdf = self.k2 * torch.bmm(J_sdf.transpose(-2, -1), J_sdf).sum(0).squeeze().cpu() / J_sdf.shape[0]
            b_sdf = -self.k2 * torch.bmm(J_sdf.transpose(-2, -1), robust_res_sdf).sum(0).squeeze().cpu() / J_sdf.shape[0]

            J_render = torch.cat([de_dsim3_render, de_dc_render], dim=-1)
            H_render = self.k1 * torch.bmm(J_render.transpose(-2, -1), J_render).sum(0).squeeze().cpu() / J_render.shape[0]
            b_render = -self.k1 * torch.bmm(J_render.transpose(-2, -1), robust_res_render).sum(0).squeeze().cpu() / J_render.shape[0]

            H = H_render + H_sdf
            H[pose_dim:pose_dim + self.code_len, pose_dim:pose_dim + self.code_len] += self.k3 * torch.eye(self.code_len)
            b = b_render + b_sdf
            b[pose_dim:pose_dim + self.code_len] -= self.k3 * z

            # Rotation regularization
            drot_dsim3 = drot_dsim3.unsqueeze(0)
            H_rot = torch.mm(drot_dsim3.transpose(-2, -1), drot_dsim3)
            b_rot = -(drot_dsim3.transpose(-2, -1) * res_rot).squeeze()
            H[:pose_dim, :pose_dim] += self.k4 * H_rot
            b[:pose_dim] -= self.k4 * b_rot
            # rot_loss = res_rot

            # add a small damping to the pose part
            H[:pose_dim, :pose_dim] += 1e0 * torch.eye(pose_dim)
            H[pose_dim-1, pose_dim-1] += self.s_damp  # add a large damping for scale
            # solve for the update vector
            dx = torch.mv(torch.inverse(H), b)
            delta_p = dx[:pose_dim]

            delta_c = dx[pose_dim:pose_dim + self.code_len]
            delta_t = exp_sim3(self.lr * delta_p)
            t_obj_cam = torch.mm(delta_t, t_obj_cam)
            latent_vector += self.lr * delta_c.cuda()

            # print("Object joint optimization: Iter %d, loss: %f, sdf loss: %f, "
            #       "render loss: %f, rotation loss: %f"
            #       % (e, loss, sdf_loss, render_loss, rot_loss))

        end = get_time()
        # print("Reconstruction takes %f seconds" % (end - start))
        t_cam_obj = torch.inverse(t_obj_cam)
        return ForceKeyErrorDict(t_cam_obj=t_cam_obj.numpy(),
                                 code=latent_vector.cpu().numpy(),
                                 is_good=True, loss=loss)

class MeshExtractorUncertain(MeshExtractor):
    def __init__(self, decoder, code_len=64, voxels_dim=64):
        super().__init__(decoder, code_len, voxels_dim)

        self.colormap = cm.get_cmap('plasma', 8)

    def extract_sdf(self,code):
        return decode_sdf(self.decoder, code, self.voxel_points)

    def extract_mesh_from_sdf(self,sdf_tensor):
        vertices, faces = convert_sdf_voxels_to_mesh(sdf_tensor.view(self.voxels_dim, self.voxels_dim, self.voxels_dim))
        vertices = vertices.astype("float32")
        faces = faces.astype("int32")
        return ForceKeyErrorDict(vertices=vertices, faces=faces)

    def vertice_to_volume(self,vertex,min,max,dim):
        max_v = np.array([1,1,1]) * max
        min_v = np.array([1,1,1]) * min
        ratio = np.divide((vertex-min_v),(max_v-min_v))
        v = dim * ratio
        return v

    def sigma_to_color(self,sigma,min=None,max=None):
        if (min is not None) and (max is not None):
            sigma_norm = (sigma-min)/(max-min)
        else:
            sigma_norm = sigma
        return self.colormap(sigma_norm)
    
    def sigma_vec_to_color_vec(self,sigma_vec,vis_abs_uncer=False):
        color_vec = np.zeros((len(sigma_vec),3))

        # percent
        if not vis_abs_uncer:
            min = np.percentile(sigma_vec, 10)
            max = np.percentile(sigma_vec, 90)

            # print("(min,max)=(%f,%f)"%(min,max))
        else:
            min = 0

            max = 0.0015

        for i,sigma in enumerate(sigma_vec):
            color = self.sigma_to_color(sigma,min,max)
            color_vec[i] = color[:3]

        return color_vec

    def extract_mesh_from_sdf_with_uncertainty(self,sdf_tensor,sdf_sigma_tensor,vis_abs_uncer=False):
        mesh = self.extract_mesh_from_sdf(sdf_tensor)

        color_vector = None
        vertices_index = None
        if not (sdf_sigma_tensor is None):
            dim = self.voxels_dim
            sdf_sigma_volume = sdf_sigma_tensor.view(dim,dim,dim)
            # add color as sigma
            # vertice to volume
            color_vector = np.zeros((len(mesh.vertices),3))
            sigma_vector = np.zeros((len(mesh.vertices),))

            vertices_index = np.zeros((len(mesh.vertices),3),np.int32)
            for i,vert in enumerate(mesh.vertices):
                v = self.vertice_to_volume(vert,-1,1,dim)

                # interpolate nearest value
                # try1: nearest value
                v_round = np.round(v).astype(np.int32)

                # make sure v_round is in [0,63]
                v_round = np.clip(v_round, 0, dim-1)

                # get sigma
                sigma = sdf_sigma_volume[v_round[0]][v_round[1]][v_round[2]]

                sigma_vector[i] = sigma

                vertices_index[i,:]=v_round

            # change to color, according to min and max of sigma
            color_vector = self.sigma_vec_to_color_vec(sigma_vector,vis_abs_uncer)

        return ForceKeyErrorDict(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=color_vector),vertices_index

    # Attention: code_sigma is code_sigma. Wait to be changed.
    def sample_codes_and_decode_sdf(self, code, code_sigma = 0.1, N = 10, input_points=None):       
        code_list = self.sample_codes(code,code_sigma,N)

        # Batch version: batch version gpus memory cost too much
        # sdf_list_ = self.decode_code_list_batch(code_list)
        sdf_list_ = self.decode_code_list(code_list, input_points)
        # t4 = time()
        sdf_mean = sdf_list_.mean(0)
        sdf_sigma = sdf_list_.var(0).sqrt()
        # t5 = time()

        # print('decode1:', t2-t1)
        # print('analy1:',t3-t2)
        # print('decode1:', t4-t3)
        # print('analy1:',t5-t4)

        return sdf_mean, sdf_sigma

    def extract_mesh_from_code_with_uncertainty(self, code, code_sigma = 0.1, N = 10, vis_abs_uncer = False):
        if code_sigma is None:
            # use no uncertainty version
            return self.extract_mesh_from_code(code)

        start = get_time()
        latent_vector = torch.from_numpy(code[:self.code_len]).cuda()
        sdf_tensor = self.extract_sdf(latent_vector)

        # generate sigma
        sdf_mean, sdf_sigma = self.sample_codes_and_decode_sdf(code, code_sigma, N)
        mesh,vertex_index = self.extract_mesh_from_sdf_with_uncertainty(sdf_tensor.view(self.voxels_dim, self.voxels_dim, self.voxels_dim),sdf_sigma, vis_abs_uncer=vis_abs_uncer)
        
        vertices = mesh.vertices.astype("float32")
        faces = mesh.faces.astype("int32")
        vertex_colors = mesh.vertex_colors
        end = get_time()
        # print("Extract mesh takes %f seconds" % (end - start))
        return ForceKeyErrorDict(vertices=vertices, faces=faces, vertex_colors=vertex_colors)

    def visualize_surface_with_probability_bound(self, code, code_sigma, sample_num=10, prob_list=[0.9974, 0.95, 0.68],
        color = [0,0,1], store_single_pts=False):
        sdf_mean, sdf_sigma = self.sample_codes_and_decode_sdf(code, code_sigma, sample_num)

        # we store points with different color/size for different prob in ONE point cloud
        if store_single_pts:
            obj_pcd = o3d.geometry.PointCloud()
            for prob in prob_list:
                sigma_coeff = prob_to_coeff(prob)
                selection = (sdf_mean).abs() < (sigma_coeff * sdf_sigma)

                valid_points = self.voxel_points[selection]

                pnum = len(valid_points)

                points_color = torch.Tensor(color).expand(pnum, 3) * prob
                obj_pcd.points.extend(o3d.utility.Vector3dVector(valid_points.cpu().numpy()))
                obj_pcd.colors.extend(o3d.utility.Vector3dVector(points_color.cpu().numpy()))

            return obj_pcd
        else:
            o3dpts_areas = []
            for prob in prob_list:
                sigma_coeff = prob_to_coeff(1-prob)
                selection = (sdf_mean).abs() < (sigma_coeff * sdf_sigma)

                valid_points = self.voxel_points[selection]

                pnum = len(valid_points)

                obj_pcd = o3d.geometry.PointCloud()
                points_color = torch.Tensor(color).expand(pnum, 3) * prob
                obj_pcd.points = o3d.utility.Vector3dVector(valid_points.cpu().numpy())
                obj_pcd.colors = o3d.utility.Vector3dVector(points_color.cpu().numpy())

                o3dpts_areas.append(ForceKeyErrorDict(pts=obj_pcd, prob=prob))

            return o3dpts_areas

    # sigma: constant for all; or a vector to specify each dimensions
    def sample_codes(self,code,sigma,N):
        code_size = code.size
        code_list = np.zeros((N,code_size))
        if len(sigma)==1:
            sigma_vec = np.zeros(code_size,) * sigma
        else:
            sigma_vec=sigma
        for i in range(N):
            gaussian_noise = np.random.normal(0.0,1,code_size)
            gaussian_noise = np.multiply(gaussian_noise,sigma_vec)
            code_sample = code + gaussian_noise
            code_list[i] = code_sample
        
        return code_list

    def decode_code_list(self, code_list, input_points=None):
        max_batch = 64**3
        ##
        N = len(code_list)
        sdf_tensor_list = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if input_points is None:
            input_points = self.voxel_points
        if not isinstance(input_points, torch.Tensor):
            input_points = torch.from_numpy(input_points)
        input_points = input_points.to(device).float()
        for code in code_list:
            # numpy to gpu
            # code = np.float(code)
            code = torch.tensor(code).cuda()
            code = code.to(torch.float32)
            sdf_tensor = decode_sdf(self.decoder, code, input_points, max_batch=max_batch) # define your own voxel_points
            sdf_tensor_list.append(sdf_tensor)
        
        tensor_sdf_all = torch.stack(sdf_tensor_list)
        return tensor_sdf_all

    def calculate_mean_sigma(self,sdf_list):
        N = len(sdf_list)
        if N >0:
            # how to calculate the mean and std in the list
            sdf_all = torch.zeros((N,len(sdf_list[0])))
            for i,sdf in enumerate(sdf_list):
                sdf_all[i] = sdf
            
            sdf_mean = torch.sum(sdf_all, dim=0) / N
            diff = (sdf_all - sdf_mean)
            sdf_sigma = torch.sqrt(torch.sum(torch.square(diff),dim=0)/N)
            return sdf_mean,sdf_sigma
        else:
            return None,None

    # input: code_sigma can be a constant, or a vector
    def generate_mesh_for_vis(self,code,code_sigma,N,vis_abs_uncer=False):
        '''
        @ code, code_sigma: numpy array
        '''
        mesh = self.extract_mesh_from_code_with_uncertainty(code,code_sigma,N,vis_abs_uncer=vis_abs_uncer)

        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        if 'vertex_colors' in mesh:
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(mesh.vertex_colors)
        mesh_o3d.compute_vertex_normals()

        return mesh_o3d

    def get_code_sigma(self,output,len_code):
        code = output[:len_code]
        alpha = output[len_code:]
        
        # change output alpha to sigma!
        sigma = torch.sqrt(torch.exp(alpha))

        code = code.cpu().numpy()
        sigma = sigma.cpu().numpy()

        if len(sigma) == 0:
            sigma = None

        return code,sigma