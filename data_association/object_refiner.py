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

import copy
import numpy as np
import open3d as o3d

def refine_table_pcd(pcd, visualize=False):
    print('refine_table_pcd')
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=500)
    largest_plane = pcd.select_by_index(inliers)

    hull, _ = largest_plane.compute_convex_hull()

    # crop the original point cloud using the contour of the largest flat surface
    obb = hull.get_oriented_bounding_box()

    obb_height = (obb.get_max_bound()[1] - obb.get_min_bound()[1]) * 0.5
    orig_bb = pcd.get_axis_aligned_bounding_box()
    orig_bb_height = orig_bb.get_max_bound()[1] - orig_bb.get_min_bound()[1]

    N = int(np.ceil(orig_bb_height / obb_height))

    new_pcd = o3d.geometry.PointCloud()
    for i in range(2*N):
        partial_pcd = pcd.crop(obb)
        new_pcd += partial_pcd
        obb.translate((0, -obb_height/2.5, 0))

    new_pcd = new_pcd.voxel_down_sample(voxel_size=0.05)

    if visualize:

        # create a visualizer object
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer()
        vis.show_settings = True
        vis.add_geometry("PCD", new_pcd)
        #vis.add_geometry("obb"+str(i), obb)

        vis.reset_camera_to_default()

        # visualize the scene
        app.add_window(vis)
        app.run()
    

    return new_pcd


