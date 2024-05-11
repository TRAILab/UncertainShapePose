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
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    transformed_source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0, 0])
    transformed_source_temp.paint_uniform_color([0, 1, 0])
    target_temp.paint_uniform_color([0, 0, 1])

    transformed_source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, transformed_source_temp, target_temp])

def registration(source, target, threshold = 0.1, trans_init=np.eye(4), draw=False):
    #target.estimate_normals(
    #    o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    use_color = False
    if not use_color:
        reg = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    else:
        reg = o3d.pipelines.registration.registration_colored_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationForColoredICP())
        
    if draw:
        draw_registration_result(source, target, reg.transformation)
    
    return  reg.fitness, reg.inlier_rmse, reg.transformation


def sm_evaluator(source, target, threshold = 0.1, trans_init=np.eye(4)):
    trans_init = np.eye(4)
    reg = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    return reg.fitness, reg.inlier_rmse