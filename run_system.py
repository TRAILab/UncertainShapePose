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

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Support headless mode for matplotlib
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

import open3d as o3d
import numpy as np
import torch
from datetime import datetime
import random
import glob, imageio
import cv2
from scipy.spatial.transform import Rotation as R

from reconstruct.utils import color_table, set_view, get_configs, get_decoder
from reconstruct.utils import set_view_mat
from reconstruct.loss_utils import get_time
from reconstruct.optimizer import Optimizer
from uncertainty.optimizer_uncertainty import MeshExtractorUncertain
from utils.visualizer import visualize_mesh_to_image, visualize_meshes_to_image
from utils.io import generate_sample_points
from utils.dataset_loader import Dataset
from utils.evo import evaluate_estimation_results, summarize_results_with_instance_frames_keep_dict
from utils.scannet_subset import ScanNetSubset
from utils.args import config_parser
from utils import SE3

from data_association.association_utils import name_to_coco_id, Observation, Object
from data_association.association_utils import * 

# @pts: numpy
def save_to_pcd(pts, filename):
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    if pts.shape[-1] == 4:
        pts = pts[:,:3]
    # change array to pcd file
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(pts)

    o3d.io.write_point_cloud(filename, scene_pcd)    

# Update: a mechanism to choose dataset and get data from it;
# save_pts: save poins to files.
def save_pcd_with_inputs_gt(pcd, surface_points, gt_local, save_dir, save_pts=False):

    inputs = surface_points
    gt = gt_local
    comp = np.asarray(pcd.points)

    fig_num = 4
    fig = plt.figure(figsize=(8*fig_num, 8))

    # input vis
    pts = inputs
    ax = fig.add_subplot(141, projection='3d')
    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]
    ax.scatter(x, y, z, color='b')
    ax.set_title('inputs')
    ax.set_box_aspect((1, 1, 1))

    pts = gt
    ax = fig.add_subplot(142, projection='3d')
    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]
    ax.scatter(x, y, z, color='r')
    ax.set_title('gt')
    ax.set_box_aspect((1, 1, 1))
    
    pts = comp
    ax = fig.add_subplot(143, projection='3d')
    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]
    ax.scatter(x, y, z, color='g')
    ax.set_title('est')
    ax.set_box_aspect((1, 1, 1))

    # overlap with comp & gt
    pts = comp
    ax = fig.add_subplot(144, projection='3d')
    ax.scatter(x, y, z, color='g')
    pts = gt
    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]
    ax.scatter(x, y, z, color='r')
    ax.set_box_aspect((1, 1, 1))

    ax.set_title('est & gt')

    fig.savefig(f'{save_dir}/pts-{i}.png')
    plt.close()


    # add a inputs & gt compare
    fig_num = 1
    fig = plt.figure(figsize=(8*fig_num, 8))

    # input vis
    pts = inputs
    ax = fig.add_subplot(111, projection='3d')
    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]
    ax.scatter(x, y, z, color='b')
    
    pts = gt
    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]
    ax.scatter(x, y, z, color='r')

    ax.set_title('inputs & gt')    
    ax.set_box_aspect((1, 1, 1))
    fig.savefig(f'{save_dir}/pts-extra-{i}.png')
    plt.close()

    if save_pts:
        save_to_pcd(inputs, f'{save_dir}/inputs-{i}.ply')
        save_to_pcd(gt, f'{save_dir}/gt-{i}.ply')
        save_to_pcd(comp, f'{save_dir}/comp-{i}.ply')


def save_output_as_deepsdf(save_dir, est_mesh_filename, gt_mesh_transformed, data_sdf, view_file, prefix):
    # if not os.path.exists(os.path.dirname(save_im_name)):
    #     os.makedirs(os.path.dirname(save_im_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_im_name = os.path.join(save_dir, prefix + '.png')
    
    visualize_mesh_to_image(est_mesh_filename, view_file, save_im_name)

    if gt_mesh_transformed is not None:
        gt_save_im_name = save_im_name[:-4] + '_gt.png'
        visualize_meshes_to_image([gt_mesh_transformed], view_file, gt_save_im_name)
        gt_save_im_name = save_im_name[:-4] + '_compare.png'
        visualize_meshes_to_image([est_mesh_filename, gt_mesh_transformed], view_file, gt_save_im_name, color=[None, [1, 0.706, 0]])

    # further save point cloud 
    if isinstance(data_sdf, list):
        data_sdf_stack = torch.cat(data_sdf, 0) # stack positive, negative
    else:
        data_sdf_stack = data_sdf
    sdf_pts = generate_sample_points(data_sdf_stack, sample_num = 10000)
    # we sample part of the points
    # sdf_pts = sdf_pts.
    gt_save_im_name = save_im_name[:-4] + '_sdf.png'
    visualize_meshes_to_image([sdf_pts], view_file, gt_save_im_name)
    gt_save_im_name = save_im_name[:-4] + '_sdf_compare.png'
    visualize_meshes_to_image([est_mesh_filename, sdf_pts], view_file, gt_save_im_name, color=[None, [1, 0.706, 0]])

def load_random_seed(seed=1):
    print('Set random seed for torch, random, numpy to:', seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_dataset(dataset_name, args, configs):
    if dataset_name == 'scannet':
        from utils.scannet import ScanNet
        dataset = ScanNet(args.sequence_dir)

    return dataset

def visualize_o3d_ins_to_image(o3d_list, view_file, save_im_name, vis, set_view_param=None):
    '''
    @set_view_param: (dist, theta, yaw)
    '''
    vis.clear_geometries()
    for ins_o3d in o3d_list:
        if ins_o3d is None:
            continue
        vis.add_geometry(ins_o3d)

    # load view from file
    if view_file is not None:
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(view_file)
        ctr.convert_from_pinhole_camera_parameters(param)
    elif set_view_param is not None:
        set_view(vis, set_view_param[0], set_view_param[1], set_view_param[2])

    '''
    Temp DEBUG TO Change Point Size
    '''
    vis.get_render_option().point_size = 10.0

    # update and save to image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_im_name)

def visualize_and_save_reconstruction_results(mesh_extractor, obj_list, frame_save_dir, 
                                    use_uncertainty=True, vis_uncertainty=True, vis=None, 
                                    vis_abs_uncer=True, t_cw_vis: np.array=None):
    '''
    A function to visualize the pose and shape estimation result of current instance.
    Save latent code, ply meshes.
    
    Output: a png file by rendering a 3D scene into a 2D view specified by t_cw_vis.
    
    '''
   
    mesh_list_world = []
    pts_list_world = []
    for obj_id, obj in enumerate(obj_list):
        if not obj.is_good:
            print('optimization fail for obj:', obj_id)
            continue

        # 1. generating mesh
        if use_uncertainty:
            code = obj.code[:,0]
            sigma = obj.code[:,1]
            
            if not vis_uncertainty:
                sigma = None # sigma is not valid.
        else:
            code = obj.code[:,0]

        try:
            if use_uncertainty:
                # with uncertainty painted
                mesh_o3d = mesh_extractor.generate_mesh_for_vis(code,code_sigma=sigma,N=10,vis_abs_uncer=vis_abs_uncer)
            else:
                # no uncertainty
                mesh = mesh_extractor.extract_mesh_from_code(code)
                mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
                mesh_o3d.compute_vertex_normals()
                mesh_o3d.paint_uniform_color(color_table[obj_id])

            # o3d.io.write_triangle_mesh(f"{save_dir}/mesh-local-{i}.obj", mesh_o3d)
            
        except:
            print('Fail to generate mesh for obj', obj_id)
            continue

        # save local mesh
        mesh_save_name = os.path.join(frame_save_dir, f'mesh-ins-{obj_id}.ply')

        # save mesh, and local pts to image
        if obj.pts_local is not None:
            pts = obj.pts_local.cpu().numpy()

            # transform to world with t_world_cam = obj.t_world_obj @ inv(obj.t_cam_obj)
            t_world_cam = obj.t_world_obj @ np.linalg.inv(obj.t_cam_obj)
            pts_world = np.dot(t_world_cam[:3,:3], pts.T).T + t_world_cam[:3,3]

            pts_pcd = o3d.geometry.PointCloud()
            pts_pcd.points = o3d.utility.Vector3dVector(pts_world)
        else:
            pts_pcd = None

        pts_list_world.append(pts_pcd)

        mesh_o3d_world = mesh_o3d.transform(obj.t_world_obj)
        mesh_list_world.append(mesh_o3d_world)

        # save latent code
        code_save_name = os.path.join(frame_save_dir, f'code-ins-{obj_id}.pt')
        torch.save(obj.code, code_save_name)

    '''
    Visualize the whole frame, including lidar points, and reconstructed meshes with GT POSE
    '''
    # Only valid for KITTI dataset. Add LiDAR point cloud
    if vis is None:
        vis = o3d.visualization.Visualizer()
    vis.clear_geometries()

    obj_index = 0
    # visualize meshes in 3D
    for mesh_o3d in mesh_list_world:
        vis.add_geometry(mesh_o3d)
        mesh_save_name = os.path.join(frame_save_dir, f'mesh-world-{obj_index}.ply')
        o3d.io.write_triangle_mesh(mesh_save_name, mesh_o3d)
        obj_index += 1

    # set view from t_wc_vis
    if t_cw_vis is not None:
        set_view_mat(vis, t_cw_vis)

    vis.poll_events()
    vis.update_renderer()

    im_path = os.path.join(frame_save_dir, f'visualization_3d.png')
    vis.capture_screen_image(im_path)


    obj_index = 0
    # visualize observation points in 3D
    for pts_pcd_world in pts_list_world:
        if pts_pcd_world is None:
            continue
        vis.add_geometry(pts_pcd_world)
        point_save_name = os.path.join(frame_save_dir, f"point-world-{obj_index}.ply")
        o3d.io.write_point_cloud(point_save_name, pts_pcd_world)
        obj_index += 1

    # set view from t_wc_vis
    if t_cw_vis is not None:
        set_view_mat(vis, t_cw_vis)

    vis.poll_events()
    vis.update_renderer()

    im_path = os.path.join(frame_save_dir, f'visualization_3d_w_pts.png')
    vis.capture_screen_image(im_path)



def visualize_intermediate_results(mesh_extractor, reconstruction_results_frame, 
                                           frame_save_dir,
                                           vis,
                                           frame_num = 1,
                                           use_uncertainty=True,
                                           vis_uncertainty=False,
                                           vis_abs_uncer=True,
                                           render_depth=False, BACKGROUND_DEPTH=9.0, mask=None, K=None,
                                           duration=5):
    '''
    This function generates meshes for each iteration, and render 3D mesh into 2D image, and finally
    generate a gif image over all the images.
    
    @ duration: the total time of the gif; 5 seconds
    '''

    for obj_id, obj in enumerate(reconstruction_results_frame):
        if not obj.is_good:
            continue

        intermediate_result = obj.intermediate # it stores sigma, not log(var)
        intermediate_code = intermediate_result['code']

        # visualize intermediate results
        num_steps = len(intermediate_code)
        
        # make a valid list from 0 to num_steps, make sure 0 and last ind is inside, 
        # and divide center equally with frame_num
        consider_indicies_list = np.linspace(0, num_steps - 1, frame_num).astype(int).tolist()

        mesh_list_steps = []
        T_oc_list_steps = []
        for ind in consider_indicies_list:
            code_dist = intermediate_code[ind]
            
            code_dist = code_dist.cpu().numpy()

            # 1. generating mesh
            if use_uncertainty:
                code = code_dist[:,0]
                sigma = code_dist[:,1]
                
                if not vis_uncertainty:
                    sigma = None # sigma is not valid.
            else:
                code = code_dist

            try:
                if use_uncertainty:
                    # with uncertainty painted
                    mesh_o3d = mesh_extractor.generate_mesh_for_vis(code,code_sigma=sigma,
                                                                    N=10,vis_abs_uncer=vis_abs_uncer)

                else:
                    # Origin codes
                    mesh = mesh_extractor.extract_mesh_from_code(code)
                    mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), 
                                                         o3d.utility.Vector3iVector(mesh.faces))
                    mesh_o3d.compute_vertex_normals()
                    mesh_o3d.paint_uniform_color(color_table[obj_id])
                # o3d.io.write_triangle_mesh(f"{save_dir}/mesh-local-{i}.obj", mesh_o3d)
            except:
                print('Fail to generate mesh for obj', obj_id)
                continue

            mesh_list_steps.append(mesh_o3d)
            T_oc_list_steps.append(intermediate_result['T_oc'][ind])

            if render_depth:
                # render depth for each obj code
                # note there are randomness, not exactly the same as the optimization process
                from reconstruct.loss import render_uncertain_depth

                mask = mask
                K = K
                t_obj_cam = np.linalg.inv(obj.t_cam_obj)

                latent_vector_distribution = code_dist

                depth, std = render_uncertain_depth(mask, K, t_obj_cam, decoder, latent_vector_distribution, 
                           num_depth_samples=50, BACKGROUND_DEPTH=BACKGROUND_DEPTH,
                           device='cuda', dtype=torch.float64)

                # save to the disk
                depth_save_name = os.path.join(frame_save_dir, f'depth-ins-{obj_id}-{t}.png')
                std_save_name = os.path.join(frame_save_dir, f'std-ins-{obj_id}-{t}.png')
                plt.imsave(depth_save_name, depth)
                plt.imsave(std_save_name, std)

        if obj.pts_local is not None:
            pts = obj.pts_local.cpu().numpy()  # points in camera frame
            pts_pcd = o3d.geometry.PointCloud()
            pts_pcd.points = o3d.utility.Vector3dVector(pts)
            # change pts size
            # pts_pcd.scale(0.01, center=pts_pcd.get_center())

            # specify color
            color_pts = np.array([0,255,0]) / 255.0 # r,g,b  dark purple
            pts_pcd.paint_uniform_color(color_pts)
        else:
            pts_pcd = None

        #########
        # visualize shape with pose, into the world frame.
        for t, mesh_o3d in enumerate(mesh_list_steps):
            T_oc = T_oc_list_steps[t]
            T_co = T_oc.inverse()

            # transform from object frame to camera frame
            mesh_o3d_cam = mesh_o3d.transform(T_co.detach().cpu().numpy()) # camera frame
            
            ind = consider_indicies_list[t]

            im_path = os.path.join(frame_save_dir, f'optimization-ins-{obj_id}-{ind}.png')

            visualize_o3d_ins_to_image([mesh_o3d_cam], None, im_path, vis, set_view_param=(0,0,0))

        # images under dir to gif
        im_list = glob.glob(os.path.join(frame_save_dir, f'optimization-ins-{obj_id}-*.png'))
        im_list = sorted(im_list, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        im_list = [imageio.v2.imread(im) for im in im_list]
        gif_path = os.path.join(frame_save_dir, f'optimization-ins-{obj_id}.gif')
        imageio.mimsave(gif_path, im_list, duration=duration)
        
        # whether to delete intermediate images
        im_list = glob.glob(os.path.join(frame_save_dir, f'optimization-ins-{obj_id}-*.png'))
        # for im in im_list:
        #     os.remove(im)

        #########

        # visualize all the meshes into gif
        for t, mesh_o3d in enumerate(mesh_list_steps):
            ind = consider_indicies_list[t]
            im_path = os.path.join(frame_save_dir, f'optimization-w-pts-{obj_id}-{ind}.png')
            visualize_o3d_ins_to_image([mesh_o3d, pts_pcd], None, im_path, vis, set_view_param=(0,0,0))

        # images under dir to gif
        im_list = glob.glob(os.path.join(frame_save_dir, f'optimization-w-pts-{obj_id}-*.png'))
        im_list = sorted(im_list, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        im_list = [imageio.v2.imread(im) for im in im_list]
        gif_path = os.path.join(frame_save_dir, f'optimization-w-pts-{obj_id}.gif')
        imageio.mimsave(gif_path, im_list, duration=duration)
        
        # delete intermediate images
        im_list = glob.glob(os.path.join(frame_save_dir, f'optimization-w-pts-{obj_id}-*.png'))
        # for im in im_list:
        #     os.remove(im)

        if render_depth:
            # add a gif for std and depth
            im_list = glob.glob(os.path.join(frame_save_dir, f'std-ins-{obj_id}-*.png'))     
            im_list = sorted(im_list, key=lambda x: int(x.split('-')[-1].split('.')[0]))
            im_list = [imageio.v2.imread(im) for im in im_list]
            gif_path = os.path.join(frame_save_dir, f'std-ins-{obj_id}.gif')
            imageio.mimsave(gif_path, im_list, duration=duration)
            im_list = glob.glob(os.path.join(frame_save_dir, f'std-ins-{obj_id}-*.png'))
            # for im in im_list:
            #     os.remove(im)

            # add for depth
            im_list = glob.glob(os.path.join(frame_save_dir, f'depth-ins-{obj_id}-*.png'))
            im_list = sorted(im_list, key=lambda x: int(x.split('-')[-1].split('.')[0]))
            im_list = [imageio.v2.imread(im) for im in im_list]
            gif_path = os.path.join(frame_save_dir, f'depth-ins-{obj_id}.gif')
            imageio.mimsave(gif_path, im_list, duration=duration)
            im_list = glob.glob(os.path.join(frame_save_dir, f'depth-ins-{obj_id}-*.png'))
            # for im in im_list:
            #     os.remove(im)

def plot_loss_curve(reconstruction_results_frame, frame_save_dir, loss_weights=None):
    '''

    @ loss_weights: a dict with {loss_name: weight, ...}

    '''
    loss_names = ['loss_sdf', 'loss_norm', 'loss_2d', 'loss']

    # define different curve shapes
    curve_shapes = ['--', '-.', ':', '-']        

    for id, obj in enumerate(reconstruction_results_frame):
        if not obj.is_good:
            continue

        plt.figure()
        ax = plt.gca()
        ax2 = ax.twinx()
        
        for loss_name in loss_names:
            loss_list = getattr(obj, loss_name+'_list')

            if loss_weights is not None:
                loss_list = [loss_weights[loss_name] * loss for loss in loss_list]
                label_name = f'{loss_name} * {loss_weights[loss_name]}'
            else:
                label_name = loss_name

            if loss_name == 'loss':
                # plot with another axis
                ax2.plot(loss_list, label=label_name, linestyle=curve_shapes[loss_names.index(loss_name)])
            else:
                # plot with the same axis
                ax.plot(loss_list, label=label_name, linestyle=curve_shapes[loss_names.index(loss_name)])
        
        plt.title(f'Loss curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        ax.legend(loc='upper right')

        ax.grid(True)

        plt.savefig(os.path.join(frame_save_dir, f'loss-obj-{id}.png'))
        plt.close()

        '''
        Debug mode, plot the losses in one figure for each iterations
        '''
        make_loss_gif = False
        if make_loss_gif:
            print('==> Debug mode, plot the losses in one figure for each iterations')
            n_all_steps = 200

            # No 3d loss
            loss_names = ['sdf_loss', 'loss_2d', 'norm_term']
            loss_names_vis = ['3D loss', '2D loss', 'Norm term']
  
            # solid dot with different color
            curve_shapes = ['-', '-', '-.']  

            for step in range(n_all_steps):
                plt.figure()
                ax = plt.gca()
                # ax2 = ax.twinx()
                
                for i,loss_name in enumerate(loss_names):
                    loss_list = getattr(obj, loss_name+'_list')

                    # only consider part of loss
                    loss_list = loss_list[:step]

                    if loss_weights is not None:
                        loss_list = [loss_weights[loss_name] * loss for loss in loss_list]
                        loss_name_vis = loss_names_vis[i]
                        label_name = f'{loss_name_vis}'
                    else:
                        label_name = loss_names_vis[i]

                    if loss_name == 'loss':
                        # plot with another axis
                        ax2.plot(loss_list, label=label_name, linestyle=curve_shapes[loss_names.index(loss_name)])
                    else:
                        # plot with the same axis
                        ax.plot(loss_list, label=label_name, linestyle=curve_shapes[loss_names.index(loss_name)])
                
                plt.title(f'Loss curve')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')

                ax.legend(loc='upper right')

                # add grid
                ax.grid(True)

                plt.savefig(os.path.join(frame_save_dir, f'loss-obj-{id}-step-{step}.png'))
                plt.close()

            '''
            Made a gif for all the loss curves
            '''
            im_list = glob.glob(os.path.join(frame_save_dir, f'loss-obj-{id}-*.png'))
            im_list = sorted(im_list, key=lambda x: int(x.split('-')[-1].split('.')[0]))
            im_list = [imageio.v2.imread(im) for im in im_list]
            gif_path = os.path.join(frame_save_dir, f'loss-obj-{id}.gif')
            
            # frame rate 24
            # duration = n_all_steps / 24.0 / 2.0
            fps = 24
            duration = 1000 * 1 / fps
            imageio.mimsave(gif_path, im_list, duration=duration)

    
def construct_scene(dataset, scene_name, frame_list=None):

    scene = Scene()
    scene.name = scene_name

    #for frame_id in range(0, num_frames, skip):
    for frame_id in frame_list:
        print("Working on frame:",frame_id)
        # load frame information
        frame = dataset.load_frame(scene_name, frame_id)

        for j in range(frame.n_bboxes):
            label = frame.labels[j]
            if not coco_id_in_intereted_classes(label) or frame.scores[j] < 0.6:
                continue
            obs = Observation(frame, j)
            frame.add_observation(obs)
        print("Number of observations:",frame.n_obs)

        if (frame.n_obs == 0):
            continue
        
        scene.add_frame(frame)

    scene.prune_overlapping_objects()
    scene.estimate_poses()

    return scene
    
def associate_obj_to_gt(dataset, scene):
    gt_obj_list = dataset.load_objects_from_scene(scene.name)
    gt_obj_pos = np.empty((0,3))
    for gt_obj in gt_obj_list:
        gt_obj_pos = np.vstack((gt_obj_pos, gt_obj['trs']['translation']))
    for obj in scene.objects:
        est_pos = obj.estimated_pose[0:3,3].T
        dist = np.linalg.norm(gt_obj_pos - est_pos, axis=1)
        gt_id = np.argmin(dist)
        min_dist = np.min(dist)
        if min_dist < 0.5:
            obj.obj_gt_id = gt_id
            print("Associating object",obj.obj_id,"to ground truth object",gt_id)
    return

def init_instance_order_list_in_scene(dataset, scene_name, args, 
                                      input=None, scene_order=None, vis=None):
    '''
    This function recoginizes the number of valid objects inside the scene, so that we could iterate over each objects.
    
    There are two options according to the applications:
    1) use groundtruth data associations. 
        This assumes we know data association in advance, and concentrate on shape and pose estimation.
        This can reveal the upperbound of the mapping accuracy (all observations are for this object).
    
    2) Automatically solve data association.
        This makes the system more complete, without relying on an assumption like 1).
        We need to consider all the frames of the scenes, initialize objects for each frame, and associate them
        with some data association algorithms.
        We give a simple implementation using minimum distance as below.
        
        TODO: We are still organizing the code for the automatic data association module.
        
    
    @ args.use_gt_association: whether to use gt association or use automatic association.

        if True:
            
            @ args.dataset_subset_package: 
            
                a dataset subset package containing instances information.
            
            @ args.obj_id
            
                Specify one object to deal with.
                
            Or:
                Consider all instances of this scene.
            
        if False:
        
            Automatically solve objects assocaition.
        

    '''

    output = {}

    if args.use_gt_association:
        # If using gt association, 
        # Use input package
        if args.dataset_subset_package is not None:
            # if package is in, use it
            ins_orders_list = input['ins_orders_list_all_scenes'][scene_order]
        else:
            # or, manually load
            if args.obj_id is None:
                ins_orders_list = dataset.load_objects_orders_from_scene_with_category(scene_name, category='chair')
            else:
                ins_orders_list = [args.obj_id]
        
    else:
        '''
        Automatically get objects assocaition with observations
        '''
        # reconstruct a scene by yourself
        scene = construct_scene(dataset, args.scene_name)
        scene.visualize_objects(vis)
        associate_obj_to_gt(dataset, scene)

        # whether to use args.obj_id to consider one specific object
        if args.obj_id is None:
            ins_orders_list = scene.get_object_indices_with_category('chair')
        else:
            ins_orders_list = [args.obj_id]

        output['scene'] = scene

    return ins_orders_list, output

def init_scene_list(args, dataset):
    output = {}
    if args.dataset_subset_package is not None:
        # load from the dataset subset package
        dataset_subset_data = torch.load(args.dataset_subset_package)
        # subset_scannet = {
        #     'scene_names': selected_scene_names,
        #     'scene_ins_list': selected_chair_indices,
        #     'n_scenes': len(selected_scene_names),
        #     'n_objects': n_selected_instances,
        #     'category': category,
        #     'version': 'v1',
        #     'description': 'A subset of ScanNet, with only chair instances. Used for debugging.',
        #     'time': time.asctime(time.localtime(time.time()))
        # }
        selected_scene_names = dataset_subset_data['scene_names']
        ins_orders_list_all_scenes = dataset_subset_data['scene_ins_list']
        scene_ins_frames_single = dataset_subset_data['scene_ins_frames_single']
        scene_ins_frames = dataset_subset_data['scene_ins_frames']
        
        print('==> load dataset subset from:', args.dataset_subset_package)
        print('  Description:', dataset_subset_data['description'])

        if args.continue_from_scene is not None:
            # start from this scene_name
            if args.continue_from_scene in selected_scene_names:
                start_scene_idx = selected_scene_names.index(args.continue_from_scene)

                selected_scene_names = selected_scene_names[start_scene_idx:]
                ins_orders_list_all_scenes = ins_orders_list_all_scenes[start_scene_idx:]
                scene_ins_frames_single = scene_ins_frames_single[start_scene_idx:]
                scene_ins_frames = scene_ins_frames[start_scene_idx:]
                
                print('==> continue from scene:', args.continue_from_scene)
            else:
                raise ValueError('continue from scene not in the dataset subset package')

        output['ins_orders_list_all_scenes'] = ins_orders_list_all_scenes

        # both single and multi-view frames should be considered
        output['scene_ins_frames_single'] = scene_ins_frames_single
        output['scene_ins_frames'] = scene_ins_frames

        output['selected_scene_names'] = selected_scene_names
        output['category_list'] = dataset_subset_data['category_list']
    else:
        if args.scene_name is None:
            # iterating over all the scenes
            val_scene_names = dataset.get_scene_name_list()
            MAX_SCENE_CONSIDERATION = 20  # debug
            SCENE_START_IDX = 0

            selected_scene_names = val_scene_names[SCENE_START_IDX:][:MAX_SCENE_CONSIDERATION]
        else:
            selected_scene_names = [args.scene_name]

    return selected_scene_names, output

def init_frame_ids_for_instance(args, LOOP_INS_ID, scene_order, 
                             scene_detail, dataset_subset=None,
                             sample_method='equal'):
    '''
    @option_select_frame: when using dataset_subset_package, choose how to sample frames from selected frames.
    
    @sample_method: when NOT using dataset_subset_package, how to sample frames for each instance.
        * equal: equally starting from 0 with the same interval to cover all the frames.
        * center: sample frames around the center frame to make sure the observation quality is high and in the center.
        
    TODO: Combine the two parameters
    
    '''
    # Select the frames for each instances
    # dataset frames selection
    # get all the observations of this instance
    if args.dataset_name != 'scannet':
        raise NotImplementedError

    view_num = args.view_num

    if args.use_gt_association:
        # get frames from package
        if args.dataset_subset_package is not None:
            all_frame_list_ins = scene_detail['scene_ins_frames']
            # besides the single view frame, we further sample frames from the multi-view frames
            all_frame_list = all_frame_list_ins[scene_order][LOOP_INS_ID]

            # option_select_frame = 'interval'  # max interval

            option_select_frame = args.option_select_frame

            if option_select_frame == 'stage_3':
                # Only 3 stage: single/sparse/dense, corresponding to 1,3,10 views;
                
                view_groups = {
                    1: [5],  
                    3: [5,3,7],
                    10: [5,0,1,2,3,4,6,7,8,9],
                }
                
                if not view_num in view_groups:
                    raise ValueError('Only support 1/3/10 views.')

                # Debug, output options:
                print(' - Select from', view_groups[view_num])
                print(' - all_frame_list:', all_frame_list)
                
                selected_frames = [all_frame_list[i] for i in view_groups[view_num]]
                
                all_frame_list = selected_frames
            else:
                # consider view num
                single_frame_list_ins = scene_detail['scene_ins_frames_single']
                single_frame_id = single_frame_list_ins[scene_order][LOOP_INS_ID][0]

                if view_num > 1:
                    '''
                    We need to make sure the first frame is the same as the single view frame.
                    For the remaining frames, we sample in a deterministic way so that each run is the same.
                    '''

                    # remove the single view frame
                    all_frame_list_no_single = [frame_id for frame_id in all_frame_list if frame_id != single_frame_id]
                    N_sample = view_num - 1

                    if option_select_frame == 'interval':
                        # sample in a deterministic way
                        all_frame_list_no_single = sorted(all_frame_list_no_single)
                        all_frame_list_no_single = all_frame_list_no_single[::len(all_frame_list_no_single)//N_sample]
                        selected_frames = all_frame_list_no_single[:N_sample]

                    elif option_select_frame == 'close': # get closest frame!
                        frame_dist = np.abs(np.array(all_frame_list_no_single) - single_frame_id)
                        # find top K minimum dis indices
                        min_dist_indices = np.argsort(frame_dist)[:N_sample]
                        selected_frames = [all_frame_list_no_single[i] for i in min_dist_indices]
                        # sort
                        selected_frames = sorted(selected_frames)
                    
                    # add the single view frame
                    all_frame_list = [single_frame_id] + selected_frames
                else:
                    all_frame_list = [single_frame_id]
                

            obs_id_list = all_frame_list
        else:
            '''
            The view is not specified.
            '''
            if args.frame_id is not None:
                obs_id_list = [args.frame_id]
            else:
                # Sample frames from all valid frames
                max_frame_num = view_num
                
                # Get the observed frames list!
                # dataset_subset = ScanNetSubset(args.sequence_dir, scene_name, LOOP_INS_ID, load_image = False)
                num_total_frames = len(dataset_subset)

                if sample_method == 'equal':
                    # equally sample max_frame_num frames from num_total_frames
                    obs_id_list = np.round(np.linspace(0, num_total_frames-1, max_frame_num)).astype(int)
                elif sample_method == 'center':
                    # sample frames around the center frame
                    # if single 
                    if max_frame_num == 1:
                        obs_id_list = [num_total_frames//2]
                    else:
                        # Only when the number of frames is one, the center sample_method is valid.
                        raise NotImplementedError
    else:
        # get all the observations of this instance
        object_obs_frames = scene.objects[obj_id].observations
        num_total_frames = len(object_obs_frames)
        print('===> number of observations of the object:', num_total_frames)

        if args.frame_id is not None:
            obs_id_list = [args.frame_id]
        else:               
            max_frame_num = 3  # for single frame, consider how many frames?

            # only randomly consider two frames
            # Replace: false to select once each
            obs_id_list = np.round(np.linspace(0, num_total_frames-1, max_frame_num)).astype(int)

    return obs_id_list

def resize_image(im, s):
    # Calculate the new dimensions
    new_dimensions = (int(im.shape[1]*s), int(im.shape[0]*s))

    # Resize the image
    resized_image = cv2.resize(im, new_dimensions, interpolation = cv2.INTER_AREA)

    return resized_image


def vis_frame_func(scene, obj_id, obs_id, recon_save_dir_prefix_frame,
                   dataset_subset=None,
                   args=None, resize_scale=1.0):
    # visualize RGB, Depth, Mask
    save_frame_im_dir = os.path.join(recon_save_dir_prefix_frame, 'input')
    os.makedirs(save_frame_im_dir, exist_ok=True)

    if scene is not None:
        rgb = scene.objects[obj_id].observations[obs_id].rgb
        depth = scene.objects[obj_id].observations[obs_id].depth
        mask = scene.objects[obj_id].observations[obs_id].mask_inflated
    else:
        frame = dataset_subset.get_one_frame(obs_id, load_image=True)  # in default, it loads mask from mask2former

        rgb = frame.rgb
        depth = frame.depth

        mask = dataset_subset._load_mask(dataset_subset.scene_name, 
                                         dataset_subset.obj_id, frame) # loading gt mask.

    # save to the disk
    rgb_save_name = os.path.join(save_frame_im_dir, f'rgb_f{obs_id}.png')
    depth_save_name = os.path.join(save_frame_im_dir, f'depth_f{obs_id}.png')
    mask_save_name = os.path.join(save_frame_im_dir, f'mask_f{obs_id}.png')

    cv2.imwrite(rgb_save_name, resize_image(rgb, resize_scale))
    plt.imsave(depth_save_name, resize_image(depth, resize_scale))

    if mask is None:
        print('Invalid mask for test evo.')
        return

    plt.imsave(mask_save_name, mask)
    
    '''
    Update: Crop RGB with Mask and Save
    '''
    # Find coordinates of non-zero (valid) pixels in the mask
    non_zero_indices = np.nonzero(mask)
    min_x, min_y = np.min(non_zero_indices, axis=1)
    max_x, max_y = np.max(non_zero_indices, axis=1)

    # Crop the area of the minimum bounding box from the RGB image
    cropped_rgb = rgb[min_x:max_x+1, min_y:max_y+1]
    cropped_rgb_save_name = os.path.join(save_frame_im_dir, f'rgb_cropped_f{obs_id}.png')
    cv2.imwrite(cropped_rgb_save_name, resize_image(cropped_rgb, resize_scale))
    
    ##
    # Crop with no Background
    # background = np.zeros_like(rgb) # Update: use white background
    background = np.ones_like(rgb) * 255
    background[mask != 0] = rgb[mask != 0]
    cropped_rgb_mask = background[min_x:max_x+1, min_y:max_y+1]
    cropped_rgb_mask_save_name = os.path.join(save_frame_im_dir, f'rgb_cropped_mask_f{obs_id}.png')
    cv2.imwrite(cropped_rgb_mask_save_name, resize_image(cropped_rgb_mask, resize_scale))
    

'''
Utils IO
'''
def construct_object_from_frame(frame, category):
    '''
    Need attributes:

        name_class
        obj_id
        
        point_cloud
        pcd_center
        
        bbox_length, bbox_height, bbox_width

    '''

    # get the idx, so that the gt mask is corresponded to the estimated mask.
    '''
    IO, only put one idx as gt mask
    '''
    frame.masks = np.stack([frame.mask])

    # Chair or tables;  Note: we use  'dining table' to propagate semantics
    if category == 'table':
        category = 'dining table'
    label_id = name_to_coco_id(category)
    frame.labels = [label_id]  # TODO: put chair label here
        
    frame.scores = [1.0]

    # idx is to select the bbox from a group of bboxes
    idx = 0
    obs = Observation(frame, idx)  

    obj_id = None
    object = Object(obj_id, obs)

    return object

def generate_gt_scannet_noise(t_wo, noise_level,
                              deterministic=True):
    # Add Gaussian Noise to the GT Pose

    t_wo_noisy = t_wo.copy()

    # get the length of diagonal line of the cuboid
    trans_wo, q_wo, s_wo = SE3.decompose_mat4(t_wo_noisy)
    # diagnal_length = np.linalg.norm(s_wo)

    sigma_trans_dis = (0.1 * s_wo) * noise_level
    sigma_rot_rad = np.pi/180.0 * 10.0 * noise_level
    sigma_scale = 0.1 * noise_level

    # sample from a gaussian distribution
    if deterministic:
        # set a random seed for np
        # the seed is from the init pose
        noise_trans_dis = sigma_trans_dis
        noise_rot_rad = sigma_rot_rad
        noise_scale = sigma_scale
    else:
        noise_trans_dis = np.random.normal(scale=sigma_trans_dis, size=3)
        noise_rot_rad = np.random.normal(scale=sigma_rot_rad, size=1)
        noise_scale = np.random.normal(scale=sigma_scale, size=3)

    # translation
    t_wo_noisy[0:3,3] += noise_trans_dis

    # Generate rotation around Z axis
    r = R.from_euler('y', noise_rot_rad, degrees=False)
    rot_per = r.as_matrix()

    t_wo_noisy[:3,:3] = np.matmul(t_wo_noisy[:3,:3], rot_per)   # right multiplication

    # scale: note x,y,z have 3 different scales
    scale_noise = 1.0 + noise_scale
    t_wo_noisy[:3, :3] = t_wo_noisy[:3, :3] * scale_noise

    out = t_wo_noisy
    
    return out

def initialize_pose(det, method = 'estimator', 
                    init_pose_estimator=None, frame=None, category=None,
                    noise_level=1.0, dataset_subset=None):
    '''
    Two options:
        gt_noise: add noise to GT pose to debug the system

        estimator: use ICP to match zero-code shape to point cloud.
    '''

    if method == 'gt_noise':

        t_world_obj = det.t_world_obj
        t_world_obj_noisy = generate_gt_scannet_noise(t_world_obj, noise_level)

        ts_world_object_init = dataset_subset.transform_obj_to_deepsdf_gt(t_world_obj_noisy)

    elif method == 'estimator':
        '''
        Construct an object class
        '''
        object = construct_object_from_frame(frame, category)

        try:
            success = init_pose_estimator.estimate(object)
        except:
            success = False

        if success:
            t_world_object_init = object.estimated_pose
            s_world_object_init = object.estimated_scale  # (3,1)

            # diag
            mat_s_world_object_init = np.diag(s_world_object_init)
            
            ts_world_object_init = t_world_object_init
            ts_world_object_init[0:3,0:3] = ts_world_object_init[0:3,0:3] @ mat_s_world_object_init

        else:   
            # fail to init
            ts_world_object_init = None

            print('! Fail to init a pose for this object.')

    # Note the output contains scale
    return ts_world_object_init

def SaveConfigsDetails(args, configs, save_dir):
    configs_dir_name = 'configs'
    configs_full_dir = os.path.join(save_dir, configs_dir_name)
    os.makedirs(configs_full_dir, exist_ok=True)
    # print args and configs into the dir
    with open(os.path.join(configs_full_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    with open(os.path.join(configs_full_dir, 'configs.txt'), 'w') as f:
        f.write(str(configs))
    
    # Save the git commit id
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        with open(os.path.join(configs_full_dir, 'git_detail.txt'), 'w') as f:
            # write all the detail git information: branch name, commit id, commit message, commit time
            save_str = f'branch: {repo.active_branch}\n'
            save_str += f'commit id: {repo.head.object.hexsha}\n'
            save_str += f'commit message: {repo.head.object.message}\n'
            save_str += f'commit time: {repo.head.object.committed_datetime}\n'
            f.write(save_str)
    except:
        print('no git')
        pass


def initialize_detections_for_frames(scene, obj_id, obs_id_list, recon_save_dir_prefix_frame, 
                                     LOOP_INS_ID, scene_name, dataset_subset, args):
    detection_hist_out = []

    fail_pack_list = []

    for obs_id in obs_id_list: # consider all the frames that observe the instance
        # print('====> Process observation', obs_id)

        '''
        Save images of the frames for debug
        '''
        vis_frame = True
        if vis_frame:
            vis_frame_func(scene, obj_id, obs_id, recon_save_dir_prefix_frame,
                   dataset_subset,
                   args, resize_scale=1.0/3.0)

        # get det of this frame
        if args.use_gt_association:
            try:
                det_list = dataset_subset.get_frame_by_id(obs_id)
                detection_hist_out+=det_list
            except:
                # if fail to load frame
                print('Fail to load frame:', obs_id)

                fail_pack = {
                    'scene_name': scene_name,
                    'ins_order': LOOP_INS_ID,
                    'sub_id': obs_id,
                    'reason': 'Fail to load get_frame_by_id',
                }
                print('Fail to reconstruct:', fail_pack)
                fail_pack_list.append(fail_pack)
        
                continue
        else:
            detection = scene.objects[obj_id].get_detection(obs_id)
            detection_hist_out.append(detection.det)
  
    return detection_hist_out, fail_pack_list

def update_configs_from_args(configs, args):
    # update weight from weight_3d, weight_2d, weight_norm
    configs.optimizer.joint_optim.k1 = args.weight_2d
    configs.optimizer.joint_optim.k2 = args.weight_3d
    configs.optimizer.joint_optim.k_norm = args.weight_norm

    # deal with close 2d/3d loss
    if args.close_2d_loss:
        print('[IMPORTANT NOTE] Close 2d loss')
        configs.optimizer.joint_optim.k1 = 0.0
    if args.close_3d_loss:
        print('[IMPORTANT NOTE] Close 3d loss')
        configs.optimizer.joint_optim.k2 = 0.0

    return

def generate_save_dir(args, configs):
    if args.cur_time is None:
        now = datetime.now()
        cur_time = now.strftime("%Y%m%d-%H%M%S")
    else:
        cur_time = args.cur_time
    
    prefix = args.prefix # exp prefix name
    dataset_name = args.dataset_name
    sample_num = args.sample_num # Sample latent codes for uncertainty
    num_iterations = args.num_iterations
    loss_type = args.loss_type

    # configs for uncertainty
    k_norm = configs.optimizer.joint_optim.k_norm
    k_2d = configs.optimizer.joint_optim.k1

    lr = args.lr
    exp_frame_name = f'{prefix}-it{num_iterations}-lr{lr}-sample{sample_num}-sig{args.init_sigma}-norm{k_norm}-k2d{k_2d}-ls-{loss_type}-ls2d-{args.loss_type_2d_uncertain}-{cur_time}/'
    save_dir = os.path.join(args.save_root, f'{dataset_name}/{exp_frame_name}')

    return save_dir

def generate_mesh_extractor(decoder, voxels_dim=64):
    mesh_extractor = MeshExtractorUncertain(decoder, voxels_dim=voxels_dim)
    return mesh_extractor    

def ProcessOneInstance(obj_id, scene_name, save_dir_scene,
                       LOOP_INS_ID, scene_order, scene_detail, 
                       optimizer, init_pose_estimator, 
                       dataset,
                       mesh_extractor,
                       vis,
                       args, configs,
                       scene=None):
    '''
    This function deal with one specific objects.
    '''
    
    ##############################################
    # Loading valid views of this instance
    ##############################################
    if args.use_gt_association:
        obj_gt_id = obj_id

        dataset_subset = ScanNetSubset(args.sequence_dir, scene_name, obj_id, load_image = False,
                                        mask_path_root=args.mask_path_root)

        num_total_frames = len(dataset_subset)
    
    else:
        # TODO: Update automatic data association
        raise NotImplementedError 
    
    # Initialize directories for each instance
    recon_save_dir_instance = os.path.join(save_dir_scene, f'ins_order_{obj_id}')
    os.makedirs(recon_save_dir_instance, exist_ok=True)

    # Get the valid observations of this instance
    obs_id_list = init_frame_ids_for_instance(args, LOOP_INS_ID, scene_order, scene_detail, dataset_subset)

    print('---> Frames list:', obs_id_list)
    
    reconstruction_results_instance = [] 
    det_id = 0
    detection_hist = []

    n_views = len(obs_id_list)

    # Use the first frame ID to initialize a saving dir
    name = f'frame-{obs_id_list[0]}'
    recon_save_dir_prefix_frame = os.path.join(recon_save_dir_instance, f'{name}')
    os.makedirs(recon_save_dir_prefix_frame, exist_ok=True)

    # For each view, initialize an observation structure
    detection_hist, fail_pack_list_init_det = initialize_detections_for_frames(
                            scene, obj_id, obs_id_list, recon_save_dir_prefix_frame, 
                            LOOP_INS_ID, scene_name, dataset_subset, args)

    # Make sure there are still valid views for multi-view fusion
    n_views_valid = len(detection_hist)
    if n_views_valid < n_views:
        print(f'Fail to get {n_views_valid}/{n_views} views for this instance. Skip it.')
        return None, fail_pack_list_init_det


    ##############################################
    # Initialize coarse pose and shape
    ##############################################
    mid_id = len(detection_hist) // 2

    init_obs_id = obs_id_list[mid_id]
    frame = dataset_subset.get_one_frame(init_obs_id, load_image=True)  
    frame.mask = dataset_subset._load_mask(dataset_subset.scene_name, dataset_subset.obj_id, frame) 
    
    # use the first frame to init pose
    init_frame_det = detection_hist[mid_id]  
    t_world_object_init = initialize_pose(init_frame_det, method = args.pose_init_method, 
                                        init_pose_estimator=init_pose_estimator, 
                                        frame=frame, category=configs.category,
                                        noise_level=args.noise_level,
                                        dataset_subset=dataset_subset)

    # Failure case to initialize a pose
    if t_world_object_init is None:
        fail_pack = {
            'scene_name': scene_name,
            'ins_order': LOOP_INS_ID,
            'sub_id': init_obs_id,
            'reason': 'Fail to init pose',
        }
        print('Fail to reconstruct:', fail_pack)

        return None, [fail_pack]

    ##############################################
    # Start pose and shape optimization
    ##############################################
    reconstruction_results_frame = []
    start = get_time()               

    try:
        obj = optimizer.reconstruct_shape_pose_world(t_world_object_init, detection_hist, \
                        code=None, sample_num=args.sample_num, loss_type=args.loss_type, num_iterations=args.num_iterations, \
                        save_intermediate=True,
                        lr=args.lr,
                        init_sigma=args.init_sigma,
                        use_2d_loss=args.use_2d_loss,
                        loss_type_2d_uncertain=args.loss_type_2d_uncertain,
                        init_sigma_pose = args.init_sigma_pose,
                        init_sigma_scale = args.init_sigma_scale,
                        render_2d_K = args.render_2d_K,
                        render_2d_calibrate_C = args.render_2d_calibrate_C,
                        render_2d_const_a = args.render_2d_const_a,
                        b_optimize_pose = not args.close_pose_optimization
                        )
        
        # Save the reconstructed results
        obj.det_id = det_id
        det_id += 1
        reconstruction_results_frame.append(obj)
    except:
        # Failure case to reconstruct
        fail_pack = {
                    'scene_name': scene_name,
                    'ins_order': obj_id,
                    'sub_id': init_obs_id,
                    'reason': 'Optimization failure'
        }
        print('Fail to reconstruct:', fail_pack)

        return None, [fail_pack]           
    
    end = get_time()
    print("Optimization costs: %f seconds" % (end - start))

    ##############################################
    # Visualization
    ##############################################

    # Visualize a 3D result into a png
    if args.open_visualization:
        # We render the 3D objects into the center view of our observations
        mid_id = len(detection_hist) // 2

        # Choose the view of visualization
        t_cw_vis = np.linalg.inv(detection_hist[mid_id]['T_world_cam'])

        visualize_and_save_reconstruction_results(mesh_extractor, reconstruction_results_frame, 
                                        frame_save_dir=recon_save_dir_prefix_frame, 
                                        vis=vis, 
                                        vis_abs_uncer=args.vis_abs_uncer, 
                                        t_cw_vis=t_cw_vis)

    # Plot the loss curve
    loss_weights = {
        'loss_sdf': configs.optimizer.joint_optim.k2,
        'loss_norm': configs.optimizer.joint_optim.k_norm, 
        'loss_2d': configs.optimizer.joint_optim.k1, 
        'loss': 1   # total loss
    }
    plot_loss_curve(reconstruction_results_frame, 
                    frame_save_dir=recon_save_dir_prefix_frame, loss_weights=loss_weights)

    # Visualize intermediate results of each iterations
    if args.open_visualization and args.visualize_intermediate:
        # Render a depth image into a specified frame
        vis_debug_render_depth = False
        if vis_debug_render_depth:
            frame_cur = dataset.get_one_frame(init_obs_id)
            K = frame_cur.K
            cam_size = frame_cur.rgb.shape[:2]
            mask = dataset._load_mask(dataset.scene_name, dataset.obj_id, frame_cur)
        else:
            mask=None
            K=None

        # Specify how many frames you want to keep from all the iterations (e.g., max of 200)
        frame_num = 10

        # fps of your gif
        fps = 24

        visualize_intermediate_results(mesh_extractor, reconstruction_results_frame, 
                                    frame_save_dir=recon_save_dir_prefix_frame,
                                    vis=vis, 
                                    frame_num = frame_num,
                                    vis_abs_uncer=args.vis_abs_uncer,
                                    render_depth=vis_debug_render_depth, 
                                    BACKGROUND_DEPTH=9.0, mask=mask, K=K,
                                    duration = 1000 * 1 / fps)   

    # TODO: This frame-level structure is redandent
    result_frame_save = {
        'sub_id': init_obs_id,
        'results': reconstruction_results_frame,
        "num_total": 1,
        "num_valid": len(reconstruction_results_frame),
        "num_views": len(obs_id_list),
        "sub_id_list": obs_id_list,
    }
    reconstruction_results_instance.append(result_frame_save)

    result_instance_save = {
        'obj_id': obj_gt_id,
        'results': reconstruction_results_instance,
        'num_total_frames': num_total_frames,
        'num_valid_frames': len(obs_id_list),
        'sub_id_list': obs_id_list
    }

    return result_instance_save, []

def ProcessOneScene(dataset, scene_name, scene_detail, scene_order,
                    init_pose_estimator, optimizer,
                    mesh_extractor, vis, 
                    args, configs, save_dir_scene):
    '''
    Deal with all the instances inside one scene.
    
    First, collect all the observations of one instance;
    Then, estimate shape and pose according to the observations;
    Finally, output the result.
    '''

    # Generate instance list; e.g., consider list of objects for chairs category
    if args.dataset_name == 'scannet':
        ins_orders_list, instance_detail = \
            init_instance_order_list_in_scene(dataset, scene_name, args, 
                                              scene_detail, scene_order, vis=vis)
    else:
        # TODO: Other dataset
        raise NotImplementedError

    print('=> instances:', ins_orders_list)

    ##############################################
    # BEGIN CONSIDERING EACH INSTANCES
    ##############################################
    reconstruction_results_scene = []

    fail_pack_list = []
    for LOOP_INS_ID, obj_id in enumerate(ins_orders_list):
        print('==> Reconstructing Instance:', obj_id)

        result_instance_save, fail_pack = ProcessOneInstance(obj_id, scene_name, save_dir_scene,
                       LOOP_INS_ID, scene_order, scene_detail, 
                       optimizer, init_pose_estimator, 
                       dataset,
                       mesh_extractor,
                       vis,
                       args, configs,
                       scene=None)
        
        if result_instance_save is not None:
            reconstruction_results_scene.append(result_instance_save)
        else:
            # check if fail_pack is a list
            if isinstance(fail_pack, list):
                fail_pack_list += fail_pack
            else:
                fail_pack_list.append(fail_pack)

    result_scene_save = {
        'scene_name': scene_name,
        'results': reconstruction_results_scene,
        'num_total_instances': len(ins_orders_list),
        'num_valid_instances': len(reconstruction_results_scene),
        'ins_orders_list': ins_orders_list
    }

    return result_scene_save, fail_pack_list

def ValidSceneCheck(scene_order, scene_detail, configs):
    '''
    Check if this scene should be processed. Conditions:
    (a) Category Check.
        The object in this scene satisfy the specific category. 
        Only used for debugs, when considering a specific category and skip others.
    '''

    if 'category_list' in scene_detail:
        # Category Check: if the instance is not corresponding to the category.
        category_list = scene_detail['category_list']
        model_category = configs.category
        obj_category = category_list[scene_order]
        if model_category != obj_category:        
            print(f'Category Differs, model: {model_category}, object: {obj_category}')
            return False
    
    return True

def ProcessAllScenes(dataset, optimizer, mesh_extractor,
                    init_pose_estimator,
                    args, configs,
                    save_dir,
                    vis = None):
    '''
    A function to process all the scenes in the dataset one by one.
    For each scenes, process all instances one by one.
    
    Inputs:
    
    @ dataset: input dataset class
    @ optimizer: optimizer
    @ mesh_extractor: uncertainty-aware shape model
    @ init_pose_estimator: dealing with initial pose
    @ vis: open3d visualizer
    
    Return:
    
    @ dataset_results: a dict including all results information
    
    '''
    
    selected_scene_names, scene_detail = initialize_dataset_information(args, dataset)
    print('=> Dataset:', args.dataset_name)        
    print('=> Consider scenes:', selected_scene_names)
        
    # record all sucessful information
    scenes_results = []
    
    # record all failure cases
    fail_pack_list = [] 

    total_scene_num = len(selected_scene_names)
    for scene_order, scene_name in enumerate(selected_scene_names):
        print('='*15)
        print('=> scene:', scene_name, f'({scene_order}/{total_scene_num})')

        save_dir_scene = os.path.join(save_dir, f'{scene_name}')

        # Check if this scene should be processed
        if not ValidSceneCheck(scene_order, scene_detail, configs):
            continue

        # Process one Scene
        result_scene_save, fail_pack_scene = ProcessOneScene(dataset, scene_name, scene_detail, scene_order,
                    init_pose_estimator, optimizer,
                    mesh_extractor, vis, 
                    args, configs, save_dir_scene)

        # Save result
        scenes_results.append(result_scene_save)

        fail_pack_list += fail_pack_scene

    dataset_results = {
        'dataset_name': args.dataset_name,
        'results': scenes_results,
        'num_total_scenes': len(selected_scene_names),
        'num_valid_scenes': len(scenes_results),
        'scene_names_list': selected_scene_names,
        
        'fail_pack_list': fail_pack_list
    }

    return dataset_results

def load_init_pose_estimator(args):
    init_pose_estimator = None
    if args.pose_init_method == 'estimator':
        '''
        We use a simple implementation for pose initialization:
            we use an average shape of a specific category (with latent=0), and use ICP matching to the observed pointcloud.
            
        Other initialization methods are also possible, e.g., learning-based 3d detector.
        '''
        from data_association.init_pose_estimator import PoseEstimator
        init_pose_estimator = PoseEstimator()        
    
    return init_pose_estimator

def initialize_dataset_information(args, dataset):
    if args.dataset_name == 'scannet':
        selected_scene_names, scene_detail = init_scene_list(args, dataset)
    elif args.dataset_name == 'KITTI':
        '''
        TODO: Consider kitti
        '''
        raise NotImplementedError

    return selected_scene_names, scene_detail


def SaveResults(result_dataset_save, 
                result_save_dir,
                args, configs, save_dir):

    SaveConfigsDetails(args, configs, save_dir)

    result_save_name = os.path.join(result_save_dir, 'result_dataset_save.pth')
    torch.save(result_dataset_save, result_save_name)
    # print('-> Save detail result information:', result_save_name)

    # save fail_pack_list
    fail_pack_save_name = os.path.join(result_save_dir, 'fail_pack_list.pth')
    torch.save(result_dataset_save['fail_pack_list'], fail_pack_save_name)
    # print('-> Save failure cases:', fail_pack_save_name)

def EvaluateResults(result_dataset_save, result_save_dir,
                    mesh_extractor,
                    args):
    
    evaluate_estimation_results(result_dataset_save, dataset_dir=args.sequence_dir, 
                                preprocessed_dir=os.path.join(args.sequence_dir, 'data/shapenet_preprocessed'),
                                result_save_dir=result_save_dir, 
                                evo_world_metric=True, mesh_extractor=mesh_extractor, mask_path_root=args.mask_path_root) # Update for scene-support

    summarize_results_with_instance_frames_keep_dict(result_dataset_save, result_save_dir=result_save_dir)

    print('Evaluation DONE.')

def run():
    ##############################################
    # Loading configurations
    ##############################################
    parser = config_parser()
    args = parser.parse_args()

    configs = get_configs(args.config)
    update_configs_from_args(configs, args)
   
    save_dir = generate_save_dir(args, configs)
    os.makedirs(save_dir, exist_ok=True)
    print('Save reconstruction result to:', save_dir)

    ##############################################
    # Init System Modules
    ##############################################
    load_random_seed(int(args.random_seed))

    # DeepSDF decoder
    decoder = get_decoder(configs)
    # Optimizer
    optimizer = Optimizer(decoder, configs)
    # MeshExtractor with uncertainty
    mesh_extractor = generate_mesh_extractor(decoder)
    # Init Pose Estimator
    init_pose_estimator = load_init_pose_estimator(args)

    # Init open3d window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)

    ##############################################
    # Load Dataset
    ##############################################
    dataset = load_dataset(args.dataset_name, args, configs)

    ##############################################
    # Process All Scenes and Instances
    ##############################################  

    dataset_results = ProcessAllScenes(dataset, optimizer, mesh_extractor,
                    init_pose_estimator,
                    args, configs,
                    save_dir,
                    vis)

    #############################
    # Save Results
    #############################
    
    print('*'*10)
    print('Save Results To:',save_dir)
    print('*'*10)

    result_save_dir = os.path.join(save_dir, 'results')
    os.makedirs(result_save_dir, exist_ok=True)
    
    SaveResults(dataset_results,  
                result_save_dir,
                args, configs, save_dir)

    #############################
    # Evaluate Results
    #############################
    print('Evaluating ...')
    EvaluateResults(dataset_results, result_save_dir,
                    mesh_extractor,
                    args)


if __name__ == "__main__":
    run()

