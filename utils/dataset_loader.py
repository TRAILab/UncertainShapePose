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

# This file is an adapter to some datasets: KITTI, ShapeNet, Pix3D and more

from reconstruct.kitti_sequence import KITIISequence

from utils.shapenet import get_dataset_loader_shapenet
from utils.shapenet_preprocessed import ShapeNetPreprocessed
from reconstruct.utils import ForceKeyErrorDict

import numpy as np
import trimesh

import open3d as o3d

class Dataset:
    def __len__(self):
        return len(self.dataset_io)


    # dataset_name: KITTI, ShapeNet, Pix3D
    def __init__(self, dataset_name, sequence_dir = None, configs = None, label = None, args = None):
        self.dataset_name = dataset_name
        if dataset_name == 'KITTI':
            self.dataset_io = KITIISequence(sequence_dir, configs)
        elif dataset_name == 'ShapeNet':
            '''
            @ ShapeNet
            the gt points are generated from sampling points from vertices
            Now it is abandoned.
            '''

            # transfer args
            dataset_type = 'test'
            dataset_dir = sequence_dir
            gt_data_dir = configs.gt_data_dir
            self.dataset_io = get_dataset_loader_shapenet(dataset_dir, dataset_type = dataset_type, gt_data_dir=gt_data_dir)

            # init gt mesh dir
        elif dataset_name == 'ShapeNetPreprocessed':
            '''
            @ ShapeNetPreprocess
            Use this one.
            '''
            split_filename = args.split_filename
            data_source = args.data_source
            # init a new simple loader to load as deepsdf.
            if 'dataset_source' in args:
                self.dataset_io = ShapeNetPreprocessed(split_filename, data_source, args.dataset_source)
            else:
                self.dataset_io = ShapeNetPreprocessed(split_filename, data_source)

        else:
            raise NotImplementedError


    # and basic function
    def get_frame_by_id(self, frame_id):

        # output a dictionary of detections.
        if self.dataset_name == 'KITTI':
            detections = self.dataset_io.get_frame_by_id(frame_id)
        elif self.dataset_name == 'ShapeNet':
            # TODO: Load detection infomation from shapenet.
            # detections = self.dataset_io.__getitem__(frame_id)

            # TODO: Load gt mesh and sample from it as points cloud directly
            # load pts from sampled gt
            gt_mesh_path = self.dataset_io.get_gt_mesh_path(frame_id)
            # load ply file
            # mesh_gt = trimesh.load(gt_mesh_path)

            ply = o3d.io.read_triangle_mesh(gt_mesh_path)
            print('gt mesh path:', gt_mesh_path)

            # normalize the mesh
            normalization_params_filename = self.dataset_io.get_normalization_path(frame_id)
            normalization_params = np.load(normalization_params_filename)
            # transform mesh
            # normalization_params["offset"], normalization_params["scale"]
            ply = ply.translate(normalization_params["offset"]).scale(normalization_params["scale"], center=(0,0,0))

            # pts_ob_sampled = ply.sample_points_uniformly(number_of_points = 2048)
            # pts_ob = np.asarray(pts_ob_sampled.points).astype(np.float32)
            pts_gt = np.asarray(ply.vertices).astype(np.float32)

            # construct a point cloud class 
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(pts_gt)
            
            # use fps
            pts_ob_sampled = scene_pcd.farthest_point_down_sample(num_samples=2048)
            pts_ob = np.asarray(pts_ob_sampled.points).astype(np.float32)

            det = ForceKeyErrorDict()
            # generate identity
            det.T_cam_obj = np.eye(4,4,dtype=np.single)
            det.surface_points = pts_ob  ### Check this line
            det.rays = None
            det.depth = None
            det.gt_local = pts_gt ### Check this line
            det.label = None
            detections = [det]

        elif self.dataset_name == 'ShapeNetPreprocessed':
            '''
            the point cloud is the same as the input of DeepSDF origin system.
            '''
            
            pts_ob = self.dataset_io.load_data(frame_id)

            det = ForceKeyErrorDict()
            # generate identity
            det.T_cam_obj = np.eye(4,4,dtype=np.single)
            det.surface_points = pts_ob  ### Check this line
            det.rays = None
            det.depth = None
            det.gt_local = self.dataset_io.load_gt_surface_sdfs(frame_id)[:,:3] ### Check this line, for visualization only!
            det.label = None
            detections = [det]

        return detections
    
    def get_current_colored_pts(self):
        if self.dataset_name == 'KITTI':
            return self.dataset_io.current_frame.get_colored_pts()
        else:
            return None, None

    def get_frame_name(self, index):
        try:
            return self.dataset_io.get_frame_name(index)
        except:
            return None