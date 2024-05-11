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
import open3d as o3d
import cv2
import copy

from scipy.optimize import linear_sum_assignment

from data_association.scan_matcher import registration, sm_evaluator
from data_association.object_refiner import *
from data_association.init_pose_estimator import *
from data_association.detection import Detection

from data_association.coco_define import COCO_METAINFO

def add_o3d_geometry(vis, name, geometry, gui=False):
    if gui:
        vis.add_geometry(name, geometry)
    else:
        vis.add_geometry(geometry)
    return

def name_to_coco_id(name):
    return COCO_METAINFO['classes'].index(name)

def coco_id_to_name(id):
    return COCO_METAINFO['classes'][id]

def coco_id_to_color(id):
    return COCO_METAINFO['palette'][id]

def coco_id_in_intereted_classes(id):
    return COCO_METAINFO['classes'][id] in COCO_METAINFO['interested_classes']

def coco_in_intereted_classes(name):
    return name in COCO_METAINFO['interested_classes']

def visualize_frame_obs(frame, coord="world"):
    # create a visualizer object
    vis = o3d.visualization.Visualizer()

    # add the point cloud and the coordinate frame to the visualizer
    vis.create_window()
    for i in range(frame.n_obs):
        if coord == "world":
            vis.add_geometry(frame.observations[i].point_cloud)
            vis.add_geometry(frame.observations[i].bbox_3d)
        elif coord == "cam":
            vis.add_geometry(frame.observations[i].point_cloud_cam)
            vis.add_geometry(frame.observations[i].bbox_cam_3d)
        else:
            raise NotImplementedError

    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])

     # visualize the scene
    vis.run()
    vis.destroy_window()

class Object:
    def __init__(self, obj_id, obs):
        self.obj_id = obj_id
        self.obj_gt_id = None
        self.label = obs.label
        self.name_class = obs.name_class
        self.coco_color = obs.coco_color
        self.score = obs.score
        self.last_seen_frame = obs.frame_id

        self.past_obs_name_classes = {}
        self.past_obs_name_classes_count = {}
        self.past_obs_name_classes[obs.name_class] = obs.score
        self.past_obs_name_classes_count[obs.name_class] = 1

        self.point_cloud = copy.copy(obs.point_cloud)
        self.bbox_3d = obs.bbox_3d
        self.bbox_height = obs.bbox_height
        self.bbox_width = obs.bbox_width
        self.bbox_length = obs.bbox_length
        self.pcd_center = obs.pcd_center
        self.bbox_center = obs.bbox_center

        if self.bbox_height is not None and self.bbox_width is not None and self.bbox_length is not None:
            self.volume = self.bbox_height * self.bbox_width * self.bbox_length

        self.n_obs = 1
        self.observations = [obs]

        self.estimated_pose = None
        self.estimated_scale = None
        self.has_est_pose = False

    def get_detection(self, obs_idx):
        return Detection(self, obs_idx)

    def add_observation(self, obs, sm_transform = None):

        if sm_transform is not None:
            obs.point_cloud.transform(sm_transform)

        point_cloud_combined = self.point_cloud + obs.point_cloud
        self.point_cloud = point_cloud_combined.voxel_down_sample(voxel_size=0.05)
        
        self.last_seen_frame = obs.frame_id
        self.n_obs += 1
        self.observations.append(obs)
        
        if obs.volume > 0.2 * self.volume:
            if obs.name_class in self.past_obs_name_classes.keys():
                self.past_obs_name_classes[obs.name_class] += obs.score
                self.past_obs_name_classes_count[obs.name_class] += 1
            else:
                self.past_obs_name_classes[obs.name_class] = obs.score
                self.past_obs_name_classes_count[obs.name_class] = 1

            if self.get_most_likely_class() == obs.name_class:
                self.label = obs.label
                self.coco_color = obs.coco_color
        self.update_class_and_score()          
        self.update_bbox()

    def merge(self, obj):
        point_cloud_combined = self.point_cloud + obj.point_cloud
        self.point_cloud = point_cloud_combined.voxel_down_sample(voxel_size=0.05)

        for past_obs_name_class in obj.past_obs_name_classes.keys():
            if past_obs_name_class in self.past_obs_name_classes.keys():
                self.past_obs_name_classes[past_obs_name_class] += obj.past_obs_name_classes[past_obs_name_class]
                self.past_obs_name_classes_count[past_obs_name_class] += obj.past_obs_name_classes_count[past_obs_name_class]
            else:
                self.past_obs_name_classes[past_obs_name_class] = obj.past_obs_name_classes[past_obs_name_class]
                self.past_obs_name_classes_count[past_obs_name_class] = obj.past_obs_name_classes_count[past_obs_name_class]

        if self.get_most_likely_class() == obj.name_class:
            self.obj_id = obj.obj_id
            self.label = obj.label
            self.coco_color = obj.coco_color
        self.update_class_and_score()

        self.last_seen_frame = max(obj.last_seen_frame, self.last_seen_frame)

        self.n_obs += obj.n_obs
        self.observations += obj.observations

        self.update_bbox()

    def get_most_likely_class(self):
        return max(self.past_obs_name_classes, key=self.past_obs_name_classes.get)
    
    def update_class_and_score(self):
        self.name_class = self.get_most_likely_class()
        self.score = self.past_obs_name_classes[self.get_most_likely_class()] / self.past_obs_name_classes_count[self.get_most_likely_class()]

    def update_bbox(self):
        self.bbox_3d = self.point_cloud.get_axis_aligned_bounding_box()
        self.pcd_center = np.mean(self.point_cloud.points, axis=0)
        self.bbox_center = self.bbox_3d.get_center()

        bbox_dim = self.bbox_3d.get_max_bound() - self.bbox_3d.get_min_bound()

        self.bbox_length = bbox_dim[0] # along x axis
        self.bbox_height = bbox_dim[1] # along y axis
        self.bbox_width  = bbox_dim[2]  # along z axis

        self.volume = self.bbox_height * self.bbox_width * self.bbox_length


class Observation:
    def __init__(self, frame, idx):
        self.mask = frame.masks[idx, :, :].astype(np.uint8)
        self.mask_inflated = cv2.dilate(self.mask, np.ones((3, 3), np.uint8), iterations=1)

        self.rgb = frame.rgb * np.expand_dims(self.mask_inflated, axis=2)
        self.depth = frame.depth * self.mask_inflated
        
        self.label = frame.labels[idx]
        self.name_class = coco_id_to_name(self.label)
        self.coco_color = coco_id_to_color(self.label)
        self.score = frame.scores[idx]
        self.frame_id = frame.frame_id

        self.K = frame.K
        self.T = frame.t_world_cam

        if 'bboxes' not in frame.__dict__:
            self.bbox = None
        else:
            self.bbox = frame.bboxes[idx,:].astype(np.int32)
        
        self.valid_pixels = np.where((self.mask_inflated > 0) & (self.depth > 0))

        self.point_cloud_cam = None
        self.point_cloud = None

        self.bbox_cam_3d = None
        self.bbox_3d = None

        self.bbox_height = None
        self.bbox_width = None
        self.bbox_length = None

        self.pcd_center_cam = None
        self.pcd_center = None

        self.bbox_center_cam = None
        self.bbox_center = None

        self.volume = None
        
        self.unproject()
        
		
    def unproject(self):
        point_cloud_cam_raw = o3d.geometry.PointCloud()

        zs = self.depth[self.valid_pixels[0], self.valid_pixels[1]] / 1000.0
        xs = (self.valid_pixels[1] - self.K[0, 2]) * zs / self.K[0, 0]
        ys = (self.valid_pixels[0] - self.K[1, 2]) * zs / self.K[1, 1]

        points = np.vstack((xs, ys, zs)).T
        colors = (self.rgb[self.valid_pixels[0], self.valid_pixels[1], :] / 255.0)[:,::-1]
        
        point_cloud_cam_raw.points = o3d.utility.Vector3dVector(points)
        point_cloud_cam_raw.colors = o3d.utility.Vector3dVector(colors)


        point_cloud_cam_ds = point_cloud_cam_raw.voxel_down_sample(voxel_size=0.05)
        #point_cloud_cam_filtered, ind = point_cloud_cam_ds.remove_statistical_outlier(nb_neighbors=50, std_ratio=3)
        #self.display_inlier_outlier(point_cloud_cam_raw, ind)

        #o3d.visualization.draw_geometries([point_cloud_cam_ds])

         # extract the largest cluster
        labels = np.array(point_cloud_cam_ds.cluster_dbscan(eps=0.12, min_points=10))
        non_negative_labels = labels[labels >= 0]

        # if there are no clusters, return
        if len(non_negative_labels) == 0:
            return

        largest_cluster_label = np.argmax(np.bincount(non_negative_labels))
        largest_cluster_mask = labels == largest_cluster_label
        self.point_cloud_cam = point_cloud_cam_ds.select_by_index(np.where(largest_cluster_mask)[0])

        self.point_cloud = copy.copy(self.point_cloud_cam)
        self.point_cloud.transform(self.T)

        # Tables can be refined by finding the largest level surface
        b_open_table_refine = True
        if b_open_table_refine and self.name_class in ["table", "dining table"]:
            try:
                refined_point_cloud = refine_table_pcd(self.point_cloud)
            except:
                refined_point_cloud = self.point_cloud

                print('Error: Fail to refine the table point cloud.')

            self.point_cloud = refined_point_cloud

        self.bbox_cam_3d = self.point_cloud_cam.get_axis_aligned_bounding_box()
        self.bbox_3d = self.point_cloud.get_axis_aligned_bounding_box()

        self.pcd_center_cam = np.mean(self.point_cloud_cam.points, axis=0)
        self.pcd_center = np.dot(self.T, np.append(self.pcd_center_cam, 1.0))[:3]

        self.bbox_center_cam = self.bbox_cam_3d.get_center()
        self.bbox_center = np.dot(self.T, np.append(self.bbox_center_cam, 1.0))[:3]

        bbox_dim = self.bbox_3d.get_max_bound() - self.bbox_3d.get_min_bound()

        self.bbox_length = bbox_dim[0] # along x axis
        self.bbox_height = bbox_dim[1] # along y axis
        self.bbox_width  = bbox_dim[2]  # along z axis

        self.volume = self.bbox_height * self.bbox_width * self.bbox_length

    def visualize_cam(self):
        o3d.visualization.draw_geometries([self.point_cloud_cam])

    def visualize_world(self):
        o3d.visualization.draw_geometries([self.point_cloud])

    '''
    def display_inlier_outlier(self, cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])
    '''

class Scene:
    def __init__(self):
        self.name = ''
        self.frames = []
        self.n_frames = 0

        self.objects = []
        self.n_objects = 0
        self.curr_obj_id = 0

        self.assoc_cost_default = 10.0
        self.assoc_cost_thresh = 0.5
        self.assoc_dist_thresh = 3.0
        self.assoc_fitness_thresh = 0.25

        self.min_obs_per_object = 2

        self.pose_estimator = PoseEstimator()


    def add_frame(self, frame):
        self.frames.append(frame)
        self.n_frames += 1
        self.associate_objects()
        #self.prune_overlapping_objects()

    def add_object(self, obs):
        new_obj = Object(self.curr_obj_id, obs)
        self.objects.append(new_obj)
        self.n_objects += 1
        self.curr_obj_id += 1

    def get_object_indices_with_category(self, name_class):
        return [i for i in range(len(self.objects)) if self.objects[i].name_class == name_class]

    def delete_object(self, obj_id):
        for obj in self.objects:
            if obj.id == obj_id:
                self.objects.remove(obj)
                self.n_objects -= 1
                break

    def associate_objects(self):

        if self.n_objects == 0:
            for i in range(self.frames[-1].n_obs):
                print("Observation {}:{} is not associated with any object".format(i, self.frames[-1].observations[i].name_class))
                self.add_object(self.frames[-1].observations[i])
            return
        
        if self.frames[-1].n_obs == 0:
            return
        
        # calculate affinity matrix
        C_mtx, transforms, fitnesses, rmses = self.calulate_affinity()
        row_ind, col_ind = linear_sum_assignment(C_mtx)

        for idx in range(len(col_ind)):
            obs_idx = row_ind[idx]
            obj_idx = col_ind[idx]
            score = C_mtx[obs_idx, obj_idx]
            obs_class = self.frames[-1].observations[obs_idx].name_class
            obj_class = self.objects[obj_idx].name_class
            if score != self.assoc_cost_default:
                print("    Observation {}:{} is associated with object {}:{} with score {}".format(obs_idx, obs_class, obj_idx, obj_class, score))
                self.objects[obj_idx].add_observation(self.frames[-1].observations[obs_idx])
            
            else:
                print("    Observation {}:{} is not associated with any object".format(obs_idx, obs_class))
                self.add_object(self.frames[-1].observations[obs_idx])
        
    def calulate_affinity(self):
        
        weight_fitness = 1.0 
        weight_rmse = 1.0
        weight_trans = 1.0

        cost_matrix_fitness = self.assoc_cost_default * np.ones((self.frames[-1].n_obs, self.n_objects))
        cost_matrix_rmse = np.zeros((self.frames[-1].n_obs, self.n_objects))
        cost_matrix_trans = np.zeros((self.frames[-1].n_obs, self.n_objects))

        transforms = {}
        fitnesses = {}
        rmses = {}
        
        for i, obs in enumerate(self.frames[-1].observations):
            for j, obj in enumerate(self.objects):
                dist = np.linalg.norm(obs.bbox_center - obj.bbox_center)
                if dist > self.assoc_dist_thresh:
                    continue
                
                fitness, rmse, trans = registration(obs.point_cloud, obj.point_cloud, 0.07, draw=False)
                trans_norm = np.linalg.norm(trans[0:3,3])
                #print("Object class:", obj.name_class)
                #print("Observation class:", obs.name_class)
                #print("Fitness: {}, RMSE: {}".format(fitness, rmse))

                if fitness < self.assoc_fitness_thresh:
                    continue

                #print("Transform:", trans)
                cost_matrix_fitness[i, j] -= fitness
                cost_matrix_rmse[i, j] = rmse - 2.0
                cost_matrix_trans[i, j] = trans_norm - 2.0

                transforms[i,j] = trans
                fitnesses[i,j] = fitness
                rmses[i,j] = rmse

        weight_matrix = weight_fitness * cost_matrix_fitness + weight_rmse * cost_matrix_rmse + weight_trans * cost_matrix_trans
        #print("weight matrix:", weight_matrix)
    
        return weight_matrix, transforms, fitnesses, rmses
    

    def calculate_overlap_volume(self, obj1, obj2):
        # get the minimum and maximum coordinates of each AABB
        obj1_min = obj1.bbox_3d.get_min_bound()
        obj1_max = obj1.bbox_3d.get_max_bound()
        obj2_min = obj2.bbox_3d.get_min_bound()
        obj2_max = obj2.bbox_3d.get_max_bound()

        # calculate the intersection volume using the formula for AABB intersection volume
        dx = min(obj1_max[0], obj2_max[0]) - max(obj1_min[0], obj2_min[0])
        dy = min(obj1_max[1], obj2_max[1]) - max(obj1_min[1], obj2_min[1])
        dz = min(obj1_max[2], obj2_max[2]) - max(obj1_min[2], obj2_min[2])
        if dx < 0 or dy < 0 or dz < 0:
            return 0
        else:
            return dx * dy * dz
        
    def delete_objects(self, obj_to_delete):
        self.objects = [obj for i, obj in enumerate(self.objects) if i not in obj_to_delete]
        self.n_objects = len(self.objects)

    def prune_overlapping_objects(self):
        # check for overlap between all pairs of objects
        obj_to_delete = []

        while True:
            obj_to_delete_iter = []
            for i in range(self.n_objects):
                if i in obj_to_delete:
                    continue
                for j in range(i+1, self.n_objects):
                    if j in obj_to_delete:
                        continue
        
                    obj1 = self.objects[i]
                    obj2 = self.objects[j]
                    overlap_volume = self.calculate_overlap_volume(obj1, obj2)

                    if overlap_volume > 0.5:
                        if obj1.volume < obj2.volume:
                            fitness, rmse = sm_evaluator(obj1.point_cloud, obj2.point_cloud, 0.08)
                        else:
                            fitness, rmse = sm_evaluator(obj2.point_cloud, obj1.point_cloud, 0.08)
    
                        if (0 < rmse < 0.08) and (fitness > 0.6):
                            if obj1.volume < obj2.volume:
                                obj2.merge(obj1)
                                obj_to_delete_iter.append(i)
                                print("    Merging object {}:{} into object {}:{} with fitness {}".format(i, obj1.name_class, j, obj2.name_class, fitness))
                            else:
                                obj1.merge(obj2)
                                obj_to_delete_iter.append(j)
                                print("    Merging object {}:{} into object {}:{} with fitness {}".format(j, obj2.name_class, i, obj1.name_class, fitness))

            obj_to_delete += obj_to_delete_iter
            if len(obj_to_delete_iter) == 0:
                break    
        self.delete_objects(obj_to_delete)


    def estimate_poses(self):
        for obj in self.objects:
            obj.has_est_pose = self.pose_estimator.estimate(obj)

    def visualize_frames(self):
        # create a visualizer object
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer()
        vis.show_settings = True

        # add the frame observations to the visualizer
        for i in range(self.n_frames):
            for j in range(self.frames[i].n_obs):
                #print("Adding Frame: {}, Observation: {}".format(i, j))
                vis.add_geometry("PCD_F_"+str(i)+"_O_"+str(j), self.frames[i].observations[j].point_cloud)
                vis.add_geometry("BOX_F_"+str(i)+"_O_"+str(j), self.frames[i].observations[j].bbox_3d)
                vis.add_3d_label(self.frames[i].observations[j].bbox_center + [0,self.frames[i].observations[j].bbox_height,0], self.frames[i].observations[j].name_class)
                #vis.add_3d_label(scene.frames[i].observations[j].pcd_center + [0,scene.frames[i].observations[j].bbox_height,0], scene.frames[i].observations[j].name_class)
        
        vis.reset_camera_to_default()

        # visualize the scene
        app.add_window(vis)
        app.run()

    def visualize_objects(self, vis=None):
        # create a visualizer object
        b_valid_gui = True
        try:
            app = o3d.visualization.gui.Application.instance
            app.initialize()
        except:
            b_valid_gui = False

        if b_valid_gui:
            vis = o3d.visualization.O3DVisualizer()
            vis.show_settings = True
        else:
            if vis is None:
                vis = o3d.visualization.Visualizer()
                vis.create_window()

        # add the frame observations to the visualizer
        for i in range(self.n_objects):
            if self.objects[i].n_obs < self.min_obs_per_object:
                continue
            obj_id = self.objects[i].obj_id
           
            #print("Adding Object: {}".format(i))
            add_o3d_geometry(vis, "PCD_OBJ_"+str(obj_id), self.objects[i].point_cloud, b_valid_gui)
            add_o3d_geometry(vis, "BOX_OBJ_"+str(obj_id), self.objects[i].bbox_3d, b_valid_gui)
            
            if self.objects[i].has_est_pose:
                add_o3d_geometry(vis, "GT_OBJ_"+str(obj_id), self.pose_estimator.model_library.get_transformed_model(self.objects[i]), b_valid_gui)
            
            if b_valid_gui:
                vis.add_3d_label(self.objects[i].bbox_center + [0,self.objects[i].bbox_height,0], self.objects[i].name_class + " ID: " + str(obj_id))

        
        '''
        In case the GUI is not working for remote server.
        '''
        if not b_valid_gui:
            # save to a local image
            vis.update_renderer()
            vis.poll_events()
            vis.capture_screen_image("./output/debug/vis_scene.png")
        else:
            vis.reset_camera_to_default()

            # visualize the scene
            app.add_window(vis)
            app.run()


