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


# The file stores visualizatio tools related to open3d, matplotlib etc.

import open3d as o3d
import os 

import matplotlib as plt
import numpy as np

from utils.io import read_o3d_mesh

def load_open3d_view(vis, view_file_name = "./view_file_deepsdf.json"):
    load_view_point = os.path.isfile(view_file_name)
    ctr = vis.get_view_control()
    # ctr.rotate(10.0, 0.0)
    if load_view_point:
        param = o3d.io.read_pinhole_camera_parameters(view_file_name)
        ctr.convert_from_pinhole_camera_parameters(param)
    else:
        print('Fail to load view from:', view_file_name)

def visualize_meshes_to_image(mesh_filename_list, view_file, save_im_name=None, vis=None, color=None):
    release_vis = False
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name='deepsdf', width=600, height=600)
        release_vis = True
    vis.clear_geometries()

    for i,mesh_filename in enumerate(mesh_filename_list):
        mesh_o3d = read_o3d_mesh(mesh_filename)
        if color is not None:
            color_cur = color[i]
            if color_cur is not None:
                mesh_o3d.paint_uniform_color(color_cur)

        if isinstance(mesh_o3d, o3d.geometry.TriangleMesh):
            mesh_o3d.compute_vertex_normals()
        vis.add_geometry(mesh_o3d)
    # set_view(vis, dist=2, theta=-np.pi/6)
    load_open3d_view(vis, view_file)
    vis.poll_events()
    vis.update_renderer()

    if save_im_name is not None:
        vis.capture_screen_image(save_im_name)

    if release_vis:
        vis.close()
    return

def visualize_mesh_to_image(mesh_filename, view_file, save_im_name=None, vis=None):
    '''
    @vis: open3d visualizer window
    @mesh_filename: path to .ply mesh

    '''    
    release_vis = False
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name='deepsdf', width=600, height=600)
        release_vis = True

    mesh_o3d = read_o3d_mesh(mesh_filename)
    mesh_o3d.compute_vertex_normals()
    vis.clear_geometries()
    vis.add_geometry(mesh_o3d)
    # set_view(vis, dist=2, theta=-np.pi/6)
    load_open3d_view(vis, view_file)
    vis.poll_events()
    vis.update_renderer()

    if save_im_name is not None:
        vis.capture_screen_image(save_im_name)

    if release_vis:
        vis.close()
    return
