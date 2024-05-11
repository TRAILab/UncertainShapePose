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

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import open3d as o3d

import scipy.stats as st
import os

def sigma_to_color(sigma,min=None,max=None):
    colormap = cm.get_cmap('plasma', 8)

    if (min is not None) and (max is not None):
        sigma_norm = (sigma-min)/(max-min)
    else:
        sigma_norm = sigma
    return colormap(sigma_norm)

def draw_pdf_calibration_plot(error, sigma, N_sample=50, save_dir='./'):
    '''
    @ error: (N, 1)
    @ sigma: (N, 1)
    '''
    total_num = len(error)
    p_gt_est = {}

    for p in np.linspace(0,1.0-1e-4,N_sample):
        c = st.norm.ppf((p+1)/2.0)
        # +- c * sigma range's prob 

        # check how many diff code lines into the range
        count_list = error < c * sigma
        num = np.sum(count_list)
        p_est = num / total_num
        p_gt_est[p] = p_est

    # now draw plots
    X = [0]
    X.extend(p_gt_est.keys())
    X.extend([1])
    Y = [0]
    Y.extend(p_gt_est.values())
    Y.extend([1])
    # fig,ax = plt.subplot(111)
    # ax.plot(X, Y)
    fig = plt.figure()
    plt.xlabel('Prob')
    plt.ylabel('Est')
    plt.plot(X,Y, marker='o', label='est')

    X=[0,1]
    Y=[0,1]
    plt.plot(X,Y, label='perfect')
    

    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir+'pdf_calib.png')
        
    return

def compute_uncertainty_error(gt_points, gen_points, sdf_sigma, save_dir='./'):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices

    # one direction: gt points -> generated
    # gen_points_kd_tree = KDTree(gen_points_sampled)
    # one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    # gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction: generated points->gt points
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    # gen_to_gt_chamfer = np.mean(np.square(two_distances))
    # error is stored in two_distances
    
    distance_error = two_distances
    distance_sigma = sdf_sigma.numpy()
    
    # analysis uncertainty and error
    # Plot a calibration plot; TODO: add histgram
    fig_num=1
    fig = plt.figure(figsize=(8*fig_num, 8))
    ax = fig.add_subplot(111)
    # input vis
    ax.scatter(distance_error, distance_sigma)
    ax.set_title('Calibration Plot error v.s. sigma')
    plt.xlabel('error')
    plt.ylabel('sigma')
    fig.savefig(f'{save_dir}.png')
    plt.close()
    
    # Draw PDF Plot
    save_dir_pdf = save_dir[:-4] + '_pdf.png'
    draw_pdf_calibration_plot(distance_error, distance_sigma, save_dir=save_dir_pdf)
    
    # output a estimated point cloud, with color as UNCERTAINTY
    color_uncertain = sigma_to_color(distance_sigma, distance_sigma.min(), distance_sigma.max())
    # output a estimated point cloud, with color as Error
    color_error = sigma_to_color(distance_error, (distance_error).min(), (distance_error).max())
    
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(gen_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(color_uncertain[:,:3])
    # save 
    o3d.io.write_point_cloud(save_dir+'_uncer.ply', obj_pcd)
    obj_pcd.colors = o3d.utility.Vector3dVector(color_error[:,:3])
    o3d.io.write_point_cloud(save_dir+'_error.ply', obj_pcd)
    return
