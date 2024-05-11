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

import open3d as o3d
import numpy as np

import os
import copy
from scipy.spatial.transform import Rotation as R

from data_association.scan_matcher import registration

class ObjectModel():
    def __init__(self, name_class, file):
        self.name_class = name_class
        self.point_cloud = o3d.io.read_point_cloud(file)
        self.point_cloud.voxel_down_sample(voxel_size=0.02)
        self.point_cloud.paint_uniform_color([0,0,1])

        self.dim = self.point_cloud.get_max_bound() - self.point_cloud.get_min_bound()
        self.length = self.dim[0]
        self.height = self.dim[1]
        self.width = self.dim[2]

    def get_transformed_model(self, scale=[1,1,1], transform=np.eye(4)):
        transformed_pcd = copy.copy(self.point_cloud)
        transformed_pcd.points = o3d.utility.Vector3dVector(np.asarray(transformed_pcd.points) * scale)
        transformed_pcd.transform(transform)
        return transformed_pcd

    def visualize(self, scale=[1,1,1], transform=np.eye(4)):
        o3d.visualization.draw_geometries([self.get_transformed_model(scale, transform)])

class ModelLibrary():

    def __init__(self, source="ours"):
        self.path = os.path.dirname(__file__) + "/object_models/"
        self.models = {}

        for file in os.listdir(self.path):
            if file.endswith(source+".ply"):
                name_class = self.parse_name_class(file.split("_")[0])
                self.models[name_class] = ObjectModel(name_class, self.path+file)

    def parse_name_class(self, name_class):
        if name_class in ["car", "truck", "bus"]:
            return "car"
        elif name_class in ["chair"]:
            return "chair"
        elif name_class in ["bowl"]:
            return "bowl"
        elif name_class in ["bottle"]:
            return "bottle"
        elif name_class in ["table", "dining table"]:
            return "table"
        elif name_class in ["sofa", "couch"]:
            return "sofa"
        else:
            return name_class
        
    def has_model(self, name_class):
        return self.parse_name_class(name_class) in self.models.keys()
    
    def get_model(self, name_class):
        return self.models[self.parse_name_class(name_class)]

    def get_transformed_model(self, name_class, scale=[1,1,1], transform=np.eye(4)):
        return self.models[self.parse_name_class(name_class)].get_transformed_model(scale, transform)
    
    def get_transformed_model(self, object):
        return self.models[self.parse_name_class(object.name_class)].get_transformed_model(object.estimated_scale, object.estimated_pose)

    def visualize(self, name_class, scale=[1,1,1], transform=np.eye(4)):
        self.models[self.parse_name_class(name_class)].visualize(scale, transform)

                
class PoseEstimator():
    def __init__(self):
        self.model_library = ModelLibrary()

    def estimate(self, object):
        object_class = object.name_class

        if not self.model_library.has_model(object_class):
            print("Object type:{} not in model library.".format(object_class))
            return False
        
        print("Estimating pose for object : {}".format(object_class))

        model = self.model_library.get_model(object_class)

        est_scale = self.estimate_scale(model, object)
        est_offset = object.pcd_center

        N_candidate = 18
        rot_candidates = [i * 2 * np.pi / N_candidate for i in range(N_candidate)]

        yaw_best = 0
        trans_best = np.zeros(3)
        rmse_best = np.inf
        fitness_best = -np.inf

        for j in range(N_candidate):
            candidate_rot = R.from_euler('y', rot_candidates[j]).as_matrix()

            T_candidate = np.eye(4)
            T_candidate[0:3, 0:3] = candidate_rot
            T_candidate[0:3, 3] = est_offset

            transformed_model = copy.copy(model.point_cloud)
            transformed_model.points = o3d.utility.Vector3dVector(np.asarray(transformed_model.points) * est_scale)

            fitness, inlier_rmse, trans_icp = registration(transformed_model, object.point_cloud, 0.1, T_candidate, False)
            rot_icp = copy.copy(trans_icp[0:3,0:3])
            euler_icp = R.from_matrix(rot_icp).as_euler('yxz')
            trans_icp = copy.copy(trans_icp[0:3,3])

            #print("candiate:", j)
            #print("fitness: ", fitness)
            #print("inlier_rmse: ", inlier_rmse)
            #print("euler_icp: ", euler_icp)

            if fitness > fitness_best:
                yaw_best = euler_icp[0]
                trans_best = trans_icp
                fitness_best = fitness
    
        print("    Best fitness:", fitness_best)
        # print("    Best fitness:", fitness_best, end=", ")
        # print("yaw:", yaw_best, end=", ")
        # print("trans:", trans_best, end=", ")
        # print("scale:", est_scale)

        T_best = np.eye(4)
        T_best[0:3, 0:3] = R.from_euler('y', yaw_best).as_matrix()
        T_best[0:3, 3] = trans_best

        #self.visualize(model, object, est_scale, T_best)

        if fitness_best > 0.25:
            object.estimated_scale = est_scale
            object.estimated_pose = T_best
            return True

        return False
    
    def estimate_scale(self, model, object):
        if model.name_class in ["table", "bed"]:
            return np.array([object.bbox_length / model.length, object.bbox_height / model.height, object.bbox_width / model.width])
        else:
            return object.bbox_height / model.height * np.ones(3)

    def visualize(self, model, object, scale=[1,1,1], transform=np.eye(4)):
        
        # create a visualizer object
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer()
        vis.show_settings = True
        vis.add_geometry("MODEL", model.get_transformed_model(scale, transform))
        vis.add_geometry("OBJECT", object.point_cloud)
  
        vis.reset_camera_to_default()

        # visualize the scene
        app.add_window(vis)
        app.run()


if __name__ == "__main__":
    model_library = ModelLibrary("ours")
    print(model_library.models.keys())
    for key in model_library.models.keys():
        model_library.visualize(key)