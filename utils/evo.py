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


import os,sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__), '..'))

from utils.scannet_subset import ScanNetSubset
from utils.scannet import ScanNet

import numpy as np

import utils.SE3 as SE3
import quaternion

import copy

import torch

import open3d as o3d

from tqdm import tqdm


# helper function to calculate difference between two quaternions 
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:                                                                                                                                                                                                                                                                                                                      
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation

def calculate_IoU(bbox1, bbox2):
    '''
    IoU of two bounding boxes on image plane.

    @ bbox: (x1, y1, x2, y2)
    '''

    # calculate the intersection area
    inter = np.maximum(0, np.minimum(bbox1[2], bbox2[2]) - np.maximum(bbox1[0], bbox2[0])) * \
            np.maximum(0, np.minimum(bbox1[3], bbox2[3]) - np.maximum(bbox1[1], bbox2[1]))
    
    # calculate the union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - inter

    iou = inter / union

    return iou


def calculate_ob_ratio_from_projection(gt_bbox_world, gt_t_world_cam, mask, K):
    '''
    Project the GT BBOX area into 2D image,
    calculate the IoU between the bbox of mask, and the projected bbox.

    @ mask: (H,W) bool, indicate the interested object only
    '''

    # project the bbox to 2D
    # bbox: (8,3);  gt_bbox_cam = gt_t_world_cam.inv @ gt_bbox_world
    gt_t_cam_world = np.linalg.inv(gt_t_world_cam)

    # transform to camera frame
    gt_bbox_cam = gt_bbox_world @ gt_t_cam_world[:3,:3].T + gt_t_cam_world[:3,3]
    
    # project to image plane, with K
    gt_bbox_2d = gt_bbox_cam @ K.T
    # normalize
    gt_bbox_2d = gt_bbox_2d / gt_bbox_2d[:,2:3]

    # 8,2 projected points into a smallest bbox
    bbox_proj = np.array([gt_bbox_2d[:,0].min(), gt_bbox_2d[:,1].min(), gt_bbox_2d[:,0].max(), gt_bbox_2d[:,1].max()])

    # get the bbox of mask
    bbox_ob = np.argwhere(mask)

    # if mask is empty
    if bbox_ob.shape[0] == 0:
        return 0.0

    bbox_ob = np.array([bbox_ob[:,1].min(), bbox_ob[:,0].min(), bbox_ob[:,1].max(), bbox_ob[:,0].max()])

    # calculate IoU
    iou = calculate_IoU(bbox_proj, bbox_ob)

    debug = False
    if debug:
        print('iou:', iou)

        # plot the process into mask image
        import cv2
        mask_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_img[mask] = [255,255,255]

        # change type to int
        bbox_ob = bbox_ob.astype(np.int32)
        bbox_proj = bbox_proj.astype(np.int32)

        gt_bbox_2d = gt_bbox_2d.astype(np.int32)

        # plot the 8 projected points, and connect to form a 3D bbox in 2D
        for i in range(8):
            mask_img = cv2.circle(mask_img, (gt_bbox_2d[i,0], gt_bbox_2d[i,1]), 1, (0,0,255), 2)
        for i in range(4):
            mask_img = cv2.line(mask_img, (gt_bbox_2d[i,0], gt_bbox_2d[i,1]), (gt_bbox_2d[i+4,0], gt_bbox_2d[i+4,1]), (0,0,255), 2)
        for i in range(4):
            mask_img = cv2.line(mask_img, (gt_bbox_2d[i,0], gt_bbox_2d[i,1]), (gt_bbox_2d[(i+1)%4,0], gt_bbox_2d[(i+1)%4,1]), (0,0,255), 2)
            mask_img = cv2.line(mask_img, (gt_bbox_2d[i+4,0], gt_bbox_2d[i+4,1]), (gt_bbox_2d[(i+1)%4+4,0], gt_bbox_2d[(i+1)%4+4,1]), (0,0,255), 2)

        # draw bbox
        mask_img = cv2.rectangle(mask_img, (bbox_ob[0], bbox_ob[1]), (bbox_ob[2], bbox_ob[3]), (0,255,0), 2)
        mask_img = cv2.rectangle(mask_img, (bbox_proj[0], bbox_proj[1]), (bbox_proj[2], bbox_proj[3]), (0,0,255), 2)

        # Put text to show iou
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask_img, 'iou: {:.3f}'.format(iou), (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # get a static variable, and +1 each time
        if not hasattr(calculate_ob_ratio_from_projection, "counter"):
            calculate_ob_ratio_from_projection.counter = 0
        else:
            calculate_ob_ratio_from_projection.counter += 1
            
        cv2.imwrite(f'./output/debug/mask_img_{calculate_ob_ratio_from_projection.counter}_iou{iou}.png', mask_img)

    return iou

'''

This method of Transforming a unit bbox into the world frame is abandoned.
The final bbox is always smaller than the real ones.

We use the new method: Fit a bounding box outside the world mesh to avoid the transformations.

'''
class Cuboids:
    @staticmethod
    def init_unit_cuboid(type='norm_deepsdf'):
        '''
        Definition from Torch3D

        @ output:  (8, 3)

        @ type:
            norm_deepsdf: A bounding box with center located in zero, and with ***diagnal*** length of 2. The fareast point to center is 1. 
                    Update: there is a ball with radius 1 in the center of the box. so the length of bbox is 2!!
            basic: A bounding box located in zero, and with edge length of 1.
        '''        
        box_corner_vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]

        if type == 'norm_deepsdf':
            # move the bbox to center, with diagnal length of 2
            box_corner_vertices = np.array(box_corner_vertices) - 0.5
            # box_corner_vertices = box_corner_vertices / np.sqrt(3) * 2
            # with length of 2!
            box_corner_vertices = box_corner_vertices * 2
        elif type == 'basic':
            # move the bbox to center, with length of 1
            box_corner_vertices = np.array(box_corner_vertices) - 0.5
        elif type == 'basic-2':
            # move the bbox to center, with length of 1/2
            box_corner_vertices = np.array(box_corner_vertices) - 0.5
            box_corner_vertices = box_corner_vertices / 2
        else:
            raise NotImplementedError

        return box_corner_vertices

    @staticmethod
    def generate_bbox_from_pose(T, type='norm_deepsdf'):
        '''
        @ input: T 4x4 matrix, with scale
        '''
        bbox = Cuboids.init_unit_cuboid(type=type)

        bbox = T[:3,:3] @ bbox.T + T[:3,3:4]

        bbox = bbox.T

        return bbox

    @staticmethod
    def calculate_IoU_in_world(T1, T2):
        '''
        @ input: T 4x4 matrix, with scale
        '''

        # generate unit bbox, then transformed by those two poses
        bbox1 = Cuboids.generate_bbox_from_pose(T1)
        bbox2 = Cuboids.generate_bbox_from_pose(T2)

        # to torch
        import torch
        bbox1 = torch.from_numpy(bbox1).float().unsqueeze(0)
        bbox2 = torch.from_numpy(bbox2).float().unsqueeze(0)

        # calculate iou in the transformed frame
        vol, iou = box3d_overlap(bbox1, bbox2)

        return iou[0]

    @staticmethod
    def generate_bbox_open3d(bbox, color=[1,0,0]):
        '''
        @ bbox: (8,3)
        @ color: (3,1) r,g,b \in [0,1]

        @ output: a line set of points and lines for bbox
        '''
        import open3d as o3d

        points = bbox
        lines = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4],
                 [0,4], [1,5], [2,6], [3,0]]
        colors = [color for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set


def chamfer_distance(point_cloud1: torch.Tensor, point_cloud2: torch.Tensor, max_batch_size: int = None) -> torch.Tensor:
    """
    Calculate the Chamfer Distance between two point clouds.
    :param point_cloud1: Point cloud 1 with shape (batch_size, npoint, 3)
    :param point_cloud2: Point cloud 2 with shape (batch_size, npoint, 3)
    :param max_batch_size: Maximum batch size to process at once to avoid running out of GPU memory
    :return: Chamfer distance with shape (batch_size,)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(point_cloud1, np.ndarray):
        point_cloud1 = torch.from_numpy(point_cloud1)
    
    if isinstance(point_cloud2, np.ndarray):
        point_cloud2 = torch.from_numpy(point_cloud2)
    
    if max_batch_size is None:
        max_batch_size = point_cloud1.shape[0]
    
    distances = []
    
    for i in range(0, point_cloud1.shape[0], max_batch_size):
        pc1 = point_cloud1[i:i+max_batch_size].to(device)
        pc2 = point_cloud2[i:i+max_batch_size].to(device)
        
        dist_matrix = torch.cdist(pc1, pc2)
        
        min_dist_12 = dist_matrix.min(dim=2)[0].mean(dim=1)
        min_dist_21 = dist_matrix.min(dim=1)[0].mean(dim=1)
        
        distance = min_dist_12 + min_dist_21
        distances.extend(distance.tolist())
    
    return distances

def fit_cuboid_to_points(points):
    '''
    @ points: (N,3) torch.Tensor

    @ return: (8,3) torch.Tensor
    '''
    max_bound = torch.max(points, axis=0).values  # x,y,z
    min_bound = torch.min(points, axis=0).values  # x,y,z

    # max_bound = max_bound.cpu().numpy()
    # min_bound = min_bound.cpu().numpy()
    
    # get the 8 vertices of the bounding box
    vertices = torch.Tensor([
        [min_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]]
    ]).to(points.device)

    return vertices


def evaluate_mesh_metric_io(gt_mesh, gt_t_world_obj, offset_no, scale_no, 
                            estimation_t_world_obj_origin, estimation_mesh_o3d_norm,
                            n_sample_pts=10000, device='cuda'):
    # sample points in origin mesh coordinate
    pts1_sampled = gt_mesh.sample_points_uniformly(number_of_points=n_sample_pts)
    pts2_sampled_norm = estimation_mesh_o3d_norm.sample_points_uniformly(number_of_points=n_sample_pts)

    pts1_sampled = torch.from_numpy(np.asarray(pts1_sampled.points)).float().unsqueeze(0).to(device)
    pts2_sampled_norm = torch.from_numpy(np.asarray(pts2_sampled_norm.points)).float().unsqueeze(0).to(device)

    # fit bbox
    bbox1 = fit_cuboid_to_points(pts1_sampled.squeeze(0))
    bbox2_norm = fit_cuboid_to_points(pts2_sampled_norm.squeeze(0))

    ###################
    # Note for estimation mesh, MUST USE estimation_t_world_obj_NORM
    ##################
    t_obj_norm_obj_origin = get_t_obj_norm_obj_origin(offset_no, scale_no)
    t_obj_origin_obj_norm = np.linalg.inv(t_obj_norm_obj_origin)
    estimation_t_world_obj_norm = estimation_t_world_obj_origin @ t_obj_origin_obj_norm

    # to gpu
    estimation_t_world_obj_norm = torch.from_numpy(estimation_t_world_obj_norm).float().to(device)
    gt_t_world_obj = torch.from_numpy(gt_t_world_obj).float().to(device)

    # transform points, and bbox to world (N, 3) @ (4,4) -> (N, 3)
    pts1_sampled_world = pts1_sampled.squeeze(0) @ gt_t_world_obj[:3,:3].T + gt_t_world_obj[:3,3]
    pts2_sampled_world = pts2_sampled_norm.squeeze(0) @ estimation_t_world_obj_norm[:3,:3].T + estimation_t_world_obj_norm[:3,3]

    # iou_error, cd_error = evaluate_mesh_metric(mesh_est_world, mesh_gt_world)
    # chamfer distance
    cd_error = chamfer_distance(pts1_sampled_world.unsqueeze(0), pts2_sampled_world.unsqueeze(0))

    # bbox to world
    bbox1_world = bbox1 @ gt_t_world_obj[:3,:3].T + gt_t_world_obj[:3,3]
    bbox2_world = bbox2_norm @ estimation_t_world_obj_norm[:3,:3].T + estimation_t_world_obj_norm[:3,3]

    # calculate iou in the transformed frame; Note: Need Pytorch3d Installation
    try:
        vol, iou = box3d_overlap(bbox1_world.unsqueeze(0).cpu(), bbox2_world.unsqueeze(0).cpu())

        iou_out = iou[0].item()
    except:
        # Please install pytorch3d for IoU calculation
        iou = np.nan

        iou_out = iou


    debug_cd_world = False
    if debug_cd_world:
        # plot the sampled points, bbox in world
        
        # save mesh1, mesh2
        # o3d.io.write_triangle_mesh('./output/debug/mesh1.ply', mesh1)
        # o3d.io.write_triangle_mesh('./output/debug/mesh2.ply', mesh2)

        # save points as o3d Pointcloud
        pts1_sampled_world = pts1_sampled_world.cpu().numpy()
        pts2_sampled_world = pts2_sampled_world.cpu().numpy()

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1_sampled_world)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts2_sampled_world)

        # o3d.io.write_point_cloud('./output/debug/pts1.ply', pcd1)
        # o3d.io.write_point_cloud('./output/debug/pts2.ply', pcd2)

        # save bbox
        bbox1_world = bbox1_world.cpu().numpy()
        bbox2_world = bbox2_world.cpu().numpy()
        bbox1_o3d = Cuboids.generate_bbox_open3d(bbox1_world, color=[0,0,1])
        bbox2_o3d = Cuboids.generate_bbox_open3d(bbox2_world, color=[1,0,0])

        # o3d.io.write_line_set('./output/debug/bbox1.ply', bbox1_o3d)
        # o3d.io.write_line_set('./output/debug/bbox2.ply', bbox2_o3d)

        # visualize by open3d
        from uncertain_shape_reconstruct_scannet import visualize_o3d_ins_to_image
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480)

        visualize_o3d_ins_to_image([pcd1, pcd2, bbox1_o3d, bbox2_o3d], view_file='view_file_deepsdf.json',
                                   save_im_name='./output/debug/vis.png', vis=vis)
        

    '''
    Optionally output mid result to calculate the observation ratio.
    '''
    mid_result = {
        'gt_bbox_world': bbox1_world.cpu().numpy(),
    }

    return iou_out, cd_error[0], mid_result


def evaluate_mesh_metric(mesh1, mesh2, n_sample_pts = 10000, device='cuda'):
    '''
    Calculate IoU and CD of meshes in the same coordinate;
    Gnerally please transform them into world coordinate.

    @ mesh: o3d mesh
    '''

    # sample points from them
    pts1_sampled = mesh1.sample_points_uniformly(number_of_points=n_sample_pts)
    pts2_sampled = mesh2.sample_points_uniformly(number_of_points=n_sample_pts)

    # get points 
    pts1_sampled = torch.from_numpy(np.asarray(pts1_sampled.points)).unsqueeze(0).to(device)
    pts2_sampled = torch.from_numpy(np.asarray(pts2_sampled.points)).unsqueeze(0).to(device)

    # chamfer distance
    cd_error = chamfer_distance(pts1_sampled, pts2_sampled)

    # fit bbox to the sampled points
    # bbox (8,3)

    bbox1 = fit_cuboid_to_points(pts1_sampled.squeeze(0))
    bbox2 = fit_cuboid_to_points(pts2_sampled.squeeze(0))

    # calculate iou in the transformed frame
    vol, iou = box3d_overlap(bbox1.unsqueeze(0), bbox2.unsqueeze(0))



    debug_cd_world = True
    if debug_cd_world:
        # plot the sampled points, bbox in world
        
        # save mesh1, mesh2
        o3d.io.write_triangle_mesh('./output/debug/mesh1.ply', mesh1)
        o3d.io.write_triangle_mesh('./output/debug/mesh2.ply', mesh2)

        # save points as o3d Pointcloud
        pts1_sampled = pts1_sampled.squeeze(0).cpu().numpy()
        pts2_sampled = pts2_sampled.squeeze(0).cpu().numpy()

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1_sampled)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts2_sampled)

        o3d.io.write_point_cloud('./output/debug/pts1.ply', pcd1)
        o3d.io.write_point_cloud('./output/debug/pts2.ply', pcd2)

        # save bbox
        bbox1 = bbox1.cpu().numpy()
        bbox2 = bbox2.cpu().numpy()
        bbox1_o3d = Cuboids.generate_bbox_open3d(bbox1, color=[0,0,1])
        bbox2_o3d = Cuboids.generate_bbox_open3d(bbox2, color=[1,0,0])

        o3d.io.write_line_set('./output/debug/bbox1.ply', bbox1_o3d)
        o3d.io.write_line_set('./output/debug/bbox2.ply', bbox2_o3d)



    return iou[0], cd_error[0]


def calculate_pose_error(T_est, T_gt, sym='None'):
    '''
    @ input: T 4x4 matrix, with scale
    @ sym: symmetry type

    Ref to the official Scan2CAD code: https://github.com/skanti/Scan2CAD/blob/master/Routines/Script/EvaluateBenchmark.py
    '''

    # t = ... # estimation
    # s = ...
    # q = ...
    t, q, s = SE3.decompose_mat4(T_est)

    # gt
    t_gt, q_gt, s_gt = SE3.decompose_mat4(T_gt)

    error_translation = np.linalg.norm(t - t_gt, ord=2)
    error_scale = 100.0*np.abs(np.mean(s/s_gt) - 1)

    # --> resolve symmetry
    if sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    else:
        error_rotation = calc_rotation_diff(q, q_gt)
        # debug output
        # print('T_est:', T_est)
        # print('q:', q)
        # print('t:', t)
        # print('s:', s)
        # print('q_gt:', q_gt)
        # print('error_rotation:', error_rotation)


    # # -> define Thresholds
    # threshold_translation = 0.2 # <-- in meter
    # threshold_rotation = 20 # <-- in deg
    # threshold_scale = 20 # <-- in %
    # # <-

    # is_valid_transformation = error_translation <= threshold_translation and error_rotation <= threshold_rotation and error_scale <= threshold_scale


    return error_translation, error_rotation, error_scale

def get_t_obj_norm_obj_origin(offset_no, scale_no):
    '''
    @ offset_no: (3,)
    @ scale_no: (3,)
    '''
    t_obj_norm_obj_origin = np.eye(4)
    t_obj_norm_obj_origin[:3,:3] = np.eye(3) * scale_no
    t_obj_norm_obj_origin[:3,3] = offset_no

    return t_obj_norm_obj_origin

def estimation_to_world(T_co_norm, offset_no, scale_no, gt_t_world_cam):
    '''
    Transofrm T_co_norm into T_wo_origin, which can directly compare to gt

    @ input: T_co_norm 4x4 matrix, with scale
    '''
    T_co_origin = T_co_norm.copy()
    T_co_origin[:3,:3] = T_co_norm[:3,:3] @ (np.eye(3) * scale_no)
    T_co_origin[:3,3] = T_co_norm[:3,3] + offset_no

    T_wo_origin = gt_t_world_cam @ T_co_origin

    return T_wo_origin

####

def draw_error_curve(T_oc_list, offset_no, scale_no, gt_t_world_cam,
                    gt_t_world_obj, sym, save_curve_name='error_curve.png'):
    n_iter = len(T_oc_list)

    error_list = []
    for it in range(n_iter):
        T_oc = T_oc_list[it].numpy()
        T_co = np.linalg.inv(T_oc)

        T_wo = estimation_to_world(T_co, offset_no, scale_no, gt_t_world_cam)

        # calculate error
        error_vec = calculate_pose_error(T_wo, gt_t_world_obj, sym)

        error_list.append(error_vec)
    
    # draw plot with error
    error_name = ['trans (m)', 'rot (deg)', 'scale (%)']
    valid_line = [0.2,20,20]

    import matplotlib
    # headless
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    error_list_arr = np.array(error_list)
    fig_num = 3
    plt.figure(figsize=(fig_num*5, 5))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.plot(error_list_arr[:,i])

        # plot valid line
        plt.plot([0,n_iter],[valid_line[i],valid_line[i]],'r--')

        plt.title(error_name[i])
        # xlabel: iteration, ylabel: error
        plt.xlabel('iteration')
        plt.ylabel('error')
    plt.savefig(save_curve_name)
    # close
    plt.close()


def get_success_rate_batch(obj_errors):
    '''
    @obj_errors: (ins_num, frame_num, 3)
    '''

    # -> define Thresholds
    threshold_translation = 0.2 # <-- in meter
    threshold_rotation = 20 # <-- in deg
    threshold_scale = 20 # <-- in %

    ####

    mask_success = obj_errors[:,:,0] <= threshold_translation 
    mask_success = mask_success & (obj_errors[:,:,1] <= threshold_rotation)
    mask_success = mask_success & (obj_errors[:,:,2] <= threshold_scale)

    success_rate = np.mean(mask_success)

    return success_rate, mask_success


def get_success_rate(obj_error_list):
    '''
    ScanNet evaluation metric:
        rate of estimations that have translation error < 0.2m, rotation error < 20deg, and scale error < 20%

    @ obj_error_list: ([ [trans, rot, scale], ... ])
    '''

    # -> define Thresholds
    threshold_translation = 0.2 # <-- in meter
    threshold_rotation = 20 # <-- in deg
    threshold_scale = 20 # <-- in %

    n_all = len(obj_error_list)

    success_list = []
    for i in range(n_all):
        error_translation, error_rotation, error_scale = obj_error_list[i]
        if error_translation <= threshold_translation and error_rotation <= threshold_rotation and error_scale <= threshold_scale:
            success_list.append(i)
    
    num_valid = len(success_list)
    success_rate = num_valid / n_all
   
    return success_rate, success_list

def evaluate_estimation_results_scene(result_scene_save, dataset_dir='data/scannet', 
                                preprocessed_dir='data/shapenet_preprocessed',
                                result_save_dir=None, dataset_name='scannet',
                                evo_world_metric=False, mesh_extractor=None, object_frame='camera',
                                mask_path_root=None):
    '''
    @result_scene_save = {
        'scene_name': args.scene_name,
        'results': reconstruction_results_scene,
        'num_total_instances'
        'num_valid_instances'
    }
        - @instance_results:
            - @frame_results: {frame_result, ...}
                - frame_result: {
                    'sub_id': sub_id,
                    'results': reconstruction_results_frame,
                    "num_total": len(dets),
                    "num_valid": len(reconstruction_results_frame),
                }

                -- reconstruction_results_frame:
                    {obj, ...}

    @preprocessed_dir: For ScanNet dataset, we need the normalization files from DeepSDF.

    @evo_world_metric: if open, IoU and Chamfer Distance in world will be calculated; Please also add the extractor.

    All evaluations happen in the world frame.
    '''
    # num_scene = len(result_scene_save)

    # dataset_name = result_scene_save['dataset_name']
    if dataset_name == 'scannet':
    # load a dataset
        dataset = ScanNet(dataset_dir)
    else:
        raise NotImplementedError

    scene_name = result_scene_save['scene_name']

    obj_error_list_scene = []
    for ins_iter_id, instance_results in enumerate(result_scene_save['results']):

        obj_id = instance_results['obj_id']

        scene_result = instance_results['results']
        num_valid_frames = len(scene_result)

        obj_error_ins_list = []  # save the evaluation of all the frames of this object
        for it_frame in range(num_valid_frames):  # Each Frame
            frame_result = scene_result[it_frame]
            sub_id = frame_result['sub_id']

            # load detection results.
            obj_results = frame_result['results']
            num_valid_objs = len(obj_results)
            
            for it_obj in range(num_valid_objs): # Each Obj
                obj = obj_results[it_obj]

                if not obj.is_good:
                    continue

                estimation_t_cam_obj = obj['t_cam_obj'] # note this one has scale information
                # estimation_t_oc = np.linalg.inv(estimation_t_cam_obj)

                # load ground truth.
                try:
                    gt_data = dataset.load_object_observations_from_scene(scene_name, obj_id, load_mesh=True, load_image=False)
                except:
                    print('Fail to load GT data, skip.')
                    continue
                gt_t_world_obj = gt_data['t_world_obj']

                norm_file_name = os.path.join(preprocessed_dir, 'NormalizationParameters', gt_data['catid_cad'], gt_data['id_cad'] + '.npz')
                normalization_params = np.load(norm_file_name)
                offset_no = normalization_params["offset"]
                scale_no = normalization_params["scale"]

                sym = gt_data['sym']
                frame = gt_data['frames'][sub_id]
                gt_t_world_cam = frame.t_world_cam

                if object_frame == 'camera':

                    estimation_t_cam_obj = obj['t_cam_obj'] # note this one has scale information
    
                    estimation_t_world_obj_origin = estimation_to_world(estimation_t_cam_obj, offset_no, 
                                                                        scale_no, gt_t_world_cam)

                    # Method 1: Compare 4x4 matrix, with scale involved
                    pose_error = calculate_pose_error(estimation_t_world_obj_origin, gt_t_world_obj, sym)

                elif object_frame == 'world':
                    estimation_t_world_obj = obj['t_world_obj']

                    # if pose is nan, then skip
                    if np.isnan(estimation_t_world_obj).any():
                        print('Skip nan pose.')
                        continue

                    estimation_t_world_obj_origin = estimation_to_world(estimation_t_world_obj, offset_no, 
                                                                        scale_no, np.eye(4))

                    pose_error = calculate_pose_error(estimation_t_world_obj_origin, gt_t_world_obj, sym)

                # draw curve of errors w.r.t iterations
                open_draw_error_curve = False
                if open_draw_error_curve:
                    save_curve_name = os.path.join(result_save_dir, scene_name, f'error_curve_ins{obj_id}_f{frame.frame_id}.png')
                    os.makedirs(os.path.dirname(save_curve_name), exist_ok=True)
                    draw_error_curve(obj['intermediate']['T_oc'], offset_no, scale_no, gt_t_world_cam,
                                        gt_t_world_obj, sym, save_curve_name)

                obj_error_ins_list.append(pose_error) # consider pose only now

                obj['evo'] = {
                    'pose_error': pose_error,
                    'success': pose_error[0] <= 0.2 and pose_error[1] <= 20 and pose_error[2] <= 20
                }

                # reconstruct a mesh from latent code
                if evo_world_metric:
                    if mesh_extractor is None:
                        raise ValueError('Please add the mesh extractor for evo_world_metric.')

                    if len(obj['code'].shape) == 1:
                        code = obj['code']
                    else:
                        code = obj['code'][:,0]

                    valid_estimation_mesh = True
                    try:
                        estimation_mesh = mesh_extractor.extract_mesh_from_code(code)

                        estimation_mesh_o3d_norm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(estimation_mesh.vertices),
                                                                        o3d.utility.Vector3iVector(estimation_mesh.faces))
                    except:
                        # incase it fails to reconstruct
                        valid_estimation_mesh = False

                    # Check if gt_mesh is valid
                    valid_gt_mesh = not isinstance(gt_data['object_mesh'], str)
                    if valid_gt_mesh and valid_estimation_mesh:
                        iou_error, cd_error, mid_result = evaluate_mesh_metric_io(gt_data['object_mesh'], gt_t_world_obj, offset_no, scale_no, 
                                estimation_t_world_obj_origin, estimation_mesh_o3d_norm)

                        # Save for calculate ob ratio
                        gt_bbox_world = mid_result['gt_bbox_world']
                    else:
                        # fail to calculate 
                        iou_error = np.nan
                        cd_error = np.nan

                    obj['evo']['iou_error'] = iou_error
                    obj['evo']['cd_error'] = cd_error


                # Further consider a observation ratio. 
                calculate_ob_ratio = False
                if calculate_ob_ratio:
                    '''
                    Project the GT BBOX to the image, and calculate the ratio of the bbox area to the mask area.
                    '''
                    if evo_world_metric:

                        if valid_gt_mesh and valid_estimation_mesh:
                            scannet_subset = ScanNetSubset(dataset_dir, scene_name, obj_id, mask_path_root=mask_path_root)
                            mask = scannet_subset._load_mask(scene_name, obj_id, frame)
                            ob_ratio = calculate_ob_ratio_from_projection(gt_bbox_world, gt_t_world_cam, mask, frame.K)
                        else:
                            ob_ratio = np.nan
                        obj['evo']['ob_ratio'] = ob_ratio
                    else:
                        # not implemented yet
                        print('Please open evo_world_metric to get GT bbox.')
                        raise NotImplementedError
                    

        obj_error_list_scene.append(obj_error_ins_list)
        
    # ignore the empty scene
    obj_error_list_scene_valid = [x for x in obj_error_list_scene if len(x) > 0]

    # output statistics analysis and then save to file
    obj_error_list_scene_arr = np.array(obj_error_list_scene_valid)

    return obj_error_list_scene_arr

def evaluate_estimation_results(result_dataset_save, dataset_dir='data/scannet', 
                                preprocessed_dir='data/shapenet_preprocessed',
                                result_save_dir=None,
                                evo_world_metric=False, mesh_extractor=None,
                                mask_path_root=None):

    dataset_results = result_dataset_save['results']


    dataset_name = result_dataset_save['dataset_name']

    obj_error_list_dataset = []
    for scene_iter_id, result_scene_save in enumerate(dataset_results):

        obj_error_list_scene = evaluate_estimation_results_scene(result_scene_save, dataset_dir, 
                                                                 preprocessed_dir, result_save_dir, dataset_name,
                                                                 evo_world_metric=evo_world_metric, mesh_extractor=mesh_extractor, object_frame='world',
                                                                 mask_path_root=mask_path_root)
        if len(obj_error_list_scene) == 0:
            continue # ignore those scene w/o valid instances
        obj_error_list_dataset.append(obj_error_list_scene)

    # output statistics analysis and then save to file
    if len(obj_error_list_dataset) > 0:
        obj_error_list_dataset_arr = np.concatenate(obj_error_list_dataset, axis=0) # (N_obj,1,3)
    else:
        obj_error_list_dataset_arr = np.zeros((0, 1, 3))

    obj_error_list_dataset_arr_flat = obj_error_list_dataset_arr

    return obj_error_list_dataset_arr_flat

def summarize_results_with_instance_frames_keep_dict(reconstruction_results_scene, result_save_dir):
    '''

    'evo' : {
        'pose_error': pose_error,
        'success': True / False
    }

    '''

    collect_data = []
    failed_list_ins = []
    failed_list_scene = []

    all_results = reconstruction_results_scene['results']
    total_ins_num = 0
    for scene_data in all_results:
        scene_results = scene_data['results']
        scene_name = scene_data['scene_name']

        ins_orders_list_valid = []
        for ins_data in scene_results:
            ins_results = ins_data['results']
            obj_id = ins_data['obj_id']
            ins_orders_list_valid.append(obj_id)

            sub_id_list_valid = []
            for frame_data in ins_results:
                frame_results = frame_data['results']
                sub_id = frame_data['sub_id']

                if len(frame_results) == 0:
                    continue

                if not frame_results[0].is_good:
                    continue

                evo = frame_results[0]['evo']   # 0: only one instance for every frame.

                # if no evo, skip
                if evo is None:
                    continue

                pose_error = evo['pose_error']
                success = evo['success']
                iou = evo['iou_error']
                cd = evo['cd_error']

                collect_data.append([scene_name, obj_id, sub_id, pose_error[0], pose_error[1], pose_error[2], success, iou, cd])

                sub_id_list_valid.append(sub_id)


        ins_orders_list = scene_data['ins_orders_list']  # TO BE CHECK IF ALL RECONSTRUCTED
        total_ins_num += len(ins_orders_list)

        if len(ins_orders_list) != len(ins_orders_list_valid):
            report = {
                'scene_name': scene_name,
                'ins_orders_list': ins_orders_list,
                'ins_orders_list_valid': ins_orders_list_valid,
                'fail_ins_num': len(ins_orders_list) - len(ins_orders_list_valid)
            }
            failed_list_scene.append(report)

    #######################
    # Save output
    #######################

    # save three list to files, as csv files
    import csv
    with open(os.path.join(result_save_dir, 'collect_data.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(collect_data)
    # print('save collect_data.csv to', result_save_dir)

    # use torch pth to save 
    import torch
    torch.save(failed_list_ins, os.path.join(result_save_dir, 'failed_list_ins.pth'))
    torch.save(failed_list_scene, os.path.join(result_save_dir, 'failed_list_scene.pth'))

    # print the failed list also as csv
    with open(os.path.join(result_save_dir, 'failed_list_ins.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if len(failed_list_ins) > 0:
            writer.writerows([failed_list_ins[0].keys()])
            writer.writerows([data.values() for data in failed_list_ins])

    with open(os.path.join(result_save_dir, 'failed_list_scene.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if len(failed_list_scene) > 0:
            writer.writerows([failed_list_scene[0].keys()])
            writer.writerows([data.values() for data in failed_list_scene])

    # print('save failed_list_ins.csv to', result_save_dir)

    ####################


    # calculate the mean and std of the pose error, and the mean of success rate, success number, fail number
    mean_pose_err, std_pose_err, mean_success_rate, success_num, \
        fail_num, mean_iou, mean_cd, std_iou, std_cd, total_count_num, \
        iou_suc_rate, cd_suc_rate = print_analyze_collect_data(collect_data)

    print('='*10)
    print('Trans (m) ↓,	Rot (deg) ↓,	Scale (%) ↓,	Success Rate (0.2/20/20)  ↑,	CD ↓')
    print(f'{mean_pose_err[0]}, {mean_pose_err[1]}, {mean_pose_err[2]}, {mean_success_rate}, {mean_cd}')
    print('='*10)


def print_analyze_collect_data(collect_data, min_ratio=-1):
    '''
    @ min_ratio: in default, we consider all obs.
    '''

    # select valid collect_data 
    if min_ratio >= 0:
        collect_data = [data for data in collect_data if data[-1] >= min_ratio]
        print('select data with ob_ratio >=', min_ratio)

    if len(collect_data) == 0:
        print('No data to analyze.')
        return None

    # calculate the mean and std of the pose error, and the mean of success rate, success number, fail number
    # collect_data_arr = np.array(collect_data)
    mean_pose_err = np.mean([data[3:6] for data in collect_data], axis=0)
    std_pose_err = np.std([data[3:6] for data in collect_data], axis=0)
    mean_success_rate = np.mean([data[6] for data in collect_data])
    success_num = np.sum([data[6] for data in collect_data])

    total_count_num = len(collect_data)
    fail_num = total_count_num - success_num

    # add iou and cd
    mean_iou = np.mean([data[7] for data in collect_data])
    mean_cd = np.mean([data[8] for data in collect_data])

    # std
    std_iou = np.std([data[7] for data in collect_data])
    std_cd = np.std([data[8] for data in collect_data])

    # Further summarize the threshold 
    # Update 0905: Further evaluate success rate with CD/IoU
    iou_thresh = [0.1,0.25,0.5,0.75]
    # # CD_thresh
    cd_thresh = [0.5,0.3,0.1]
    iou_list = np.array([data[7] for data in collect_data])
    cd_list = np.array([data[8] for data in collect_data])   

    iou_suc_rate = {}
    cd_suc_rate = {}
    for iou_t in iou_thresh:
        iou_suc_rate[iou_t] = np.sum(iou_list >= iou_t) / len(collect_data)
    for cd_t in cd_thresh:
        cd_suc_rate[cd_t] = np.sum(cd_list <= cd_t) / len(collect_data)

    return mean_pose_err, std_pose_err, mean_success_rate, success_num, fail_num, mean_iou, mean_cd, std_iou, std_cd, total_count_num, \
        iou_suc_rate, cd_suc_rate


def summarize_results_with_instance_frames(obj_errors, result_save_dir):
    '''
    @obj_errors: numpy array of shape (num_instances, num_frames, 3); trans, rot, scale

    @output: a csv table to store the detail information
    '''

    # TODO: add a csv table to store the detail information

    # get success rate
    success_rate, success_mask = get_success_rate_batch(obj_errors)

    num_ins = obj_errors.shape[0]
    num_frame_per_ins = obj_errors.shape[1]
    
    num_all_obs = num_ins * num_frame_per_ins
    num_success_obs = np.sum(success_mask)

    print('success rate:', success_rate, ' (', num_success_obs, '/', num_all_obs,')')

    # combine the dims of instance and frames
    obj_errors_flatten = obj_errors.reshape(-1,3)
    print('mean of trans/rot/scale (m/deg/%):', np.mean(obj_errors_flatten, axis=0))
    print('std of trans/rot/scale (m/deg/%):', np.std(obj_errors_flatten, axis=0))

    # save to file
    if result_save_dir is not None:
        np.save(os.path.join(result_save_dir, 'obj_errors.npy'), obj_errors)
        print('save obj_errors.npy to', result_save_dir)
    
    # output a csv files storing the detail information
    # INS_ID, FRAME_ID, TRANS, ROT, SCALE, SUCCESS
    csv_list = []
    for ins_id in range(num_ins):
        for frame_id in range(num_frame_per_ins):
            row_data = [ins_id, frame_id] + obj_errors[ins_id, frame_id].tolist() + [success_mask[ins_id, frame_id]]
            csv_list.append(row_data)
    # save to csv file
    import csv
    with open(os.path.join(result_save_dir, 'obj_errors.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_list)
    print('save obj_errors.csv to', result_save_dir)
    


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_source', type=str, default=None, help='path to config file')
    return parser    

def test_multiple_runs():
    '''
    Support consider several runs together.
    Those runs may be interapted by bugs.
    '''

    import torch
    from reconstruct.utils import color_table, set_view, get_configs, get_decoder
    from reconstruct.optimizer import Optimizer, MeshExtractor

    # parse args, if no args then load 
    parser = config_parser()
    args = parser.parse_args()
    if args.input_source is None:
        source_list = _load_source_list()
    else:
        source_list = [args.input_source]
        source_list = [os.path.join(x, 'results/result_dataset_save.pth') for x in source_list]

    print('source num:', len(source_list))

    # init a mesh_extractor
    config = 'configs/config_scannet.json'
    voxels_dim = 64

    configs = get_configs(config)
    decoder = get_decoder(configs)
    mesh_extractor = MeshExtractor(decoder, voxels_dim=voxels_dim)

    obj_error_list_dataset_arr_flat_list = []
    for source_name in source_list:
        print('source:', source_name)
        reconstruction_results_scene = torch.load(source_name)

        # get dir of reconstruction_results_scene
        result_save_dir = os.path.dirname(source_name)
        obj_error_list_dataset_arr_flat = evaluate_estimation_results(reconstruction_results_scene, result_save_dir=result_save_dir,
                                                                        evo_world_metric=True, mesh_extractor=mesh_extractor)

        obj_error_list_dataset_arr_flat_list.append(obj_error_list_dataset_arr_flat)


    obj_error_list_dataset_arr_flat_list_arr = np.concatenate(obj_error_list_dataset_arr_flat_list, axis=0)

    # save to the final one
    # summarize_results_with_instance_frames(obj_error_list_dataset_arr_flat_list_arr, result_save_dir=result_save_dir)

    # save reconstruction_results_scene with evo attached to each observations
    torch.save(reconstruction_results_scene, os.path.join(result_save_dir, 'result_dataset_save_w_evo.pth'))

    # a new branch, directly extract evo results from origin dicts. So we can track the scene/instance/frame information.
    summarize_results_with_instance_frames_keep_dict(reconstruction_results_scene, result_save_dir=result_save_dir)


if __name__ == '__main__':

    test_multiple_runs()

