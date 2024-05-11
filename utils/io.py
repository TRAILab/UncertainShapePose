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


import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

# This function is refered from : [get_frame_by_id] of utils/dataset_loader.py
def sample_from_gt_mesh(gt_mesh_path, normalization_params_filename, num_samples = 2048, method = 'pointcloud'):
    ply = o3d.io.read_triangle_mesh(gt_mesh_path)
    normalization_params = np.load(normalization_params_filename)

    ply = ply.translate(normalization_params["offset"]).scale(normalization_params["scale"], center=(0,0,0))

    if method == 'vertex':
        pts_gt = np.asarray(ply.vertices).astype(np.float32)
        # construct a point cloud class 
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(pts_gt)
        
        if num_samples < len(pts_gt):
            pts_ob_sampled = scene_pcd.farthest_point_down_sample(num_samples=num_samples)
        else:
            pts_ob_sampled = scene_pcd
    elif method == 'pointcloud':
        pts_ob_sampled = ply.sample_points_uniformly(number_of_points=num_samples)
    else:
        raise NotImplementedError

    pts_ob = np.asarray(pts_ob_sampled.points).astype(np.float32)

    # concat the sdf value with 0
    sdf_values = np.zeros((pts_ob.shape[0],1))
    pts_sdf = np.concatenate([pts_ob, sdf_values], -1)

    return pts_sdf

def read_o3d_mesh(mesh_filename):
    if isinstance(mesh_filename, o3d.geometry.TriangleMesh) or isinstance(mesh_filename, o3d.geometry.PointCloud):
        mesh_o3d = mesh_filename
    else:
        # Support two types
        if '.ply' in mesh_filename:
            mesh_o3d = o3d.io.read_triangle_mesh(mesh_filename)
        elif '.obj' in mesh_filename:
            mesh_o3d = o3d.io.read_triangle_mesh(mesh_filename)
    
    return mesh_o3d

def load_gt_mesh_with_transform(gt_mesh_filename, normalization_params_filename):
    param = np.load(normalization_params_filename)
    mesh = read_o3d_mesh(gt_mesh_filename)
    mesh = mesh.translate(param['offset']).scale(param['scale'], center=(0,0,0))
    return mesh

def generate_sample_points(data_sdf, sample_num = None):
    num_sdf = len(data_sdf)
    if sample_num is not None and num_sdf > sample_num:
        indices_random = np.random.choice(num_sdf, sample_num, replace=False)
        data_sdf = data_sdf[indices_random,:]

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(data_sdf[:,:3])

    color_bar = plt.get_cmap('viridis')
    colors = color_bar(data_sdf[:,-1])
    pts.colors = o3d.utility.Vector3dVector(colors[:,:3])


    # color to sdf using a color bar.
    return pts


