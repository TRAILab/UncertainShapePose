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
from reconstruct.utils import set_view

import os
import wandb

from uncertainty.wandb_io import wandb_vis_image
from utils.visualizer import load_open3d_view

def visualize_intermediate_optimization(mesh_extractor, intermediate, pts_local, view_file, save_dir='./', jump=1, vis_uncertainty=False):
    os.makedirs(save_dir, exist_ok=True)
    # view config
    view_dist=2

    # create new window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='deepsdf', width=600, height=600)

    # visualize local points
    # pts_pcd = o3d.geometry.PointCloud()
    # pts_pcd.points = o3d.utility.Vector3dVector(pts_local.cpu().numpy()[:,:3])

    data_list = [0]
    columns=["Obj ID"]
    for i,dist in enumerate(intermediate):
        if not (i % jump == 0):
            continue
        code = dist[:,0].cpu().numpy()
        sigma = dist[:,1].cpu().numpy()
        # with uncertainty painted
        try:
            if not vis_uncertainty:
                sigma = None
            mesh_o3d = mesh_extractor.generate_mesh_for_vis(code,code_variance=sigma,N=10,vis_abs_uncer=True)

            # visualize local mesh
            vis.clear_geometries()
            # vis.add_geometry(pts_pcd)
            vis.add_geometry(mesh_o3d)
            opt = vis.get_render_option()
            opt.show_coordinate_frame = True

            # set view angles
            # pi = 3.14159
            # theta = -pi*5/6.
            # set_view(vis, dist=view_dist, theta=theta)        
            # load view file
            load_open3d_view(vis, view_file)

            # save images
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_dir+f'/it-{i}.png')
            # log to wandb
            # wandb_vis_image(f'it-{i}', save_dir+f'/it-{i}.png')
            columns.append(f'it-{i}')
            data_list.append(wandb.Image(save_dir+f'/it-{i}.png'))

        except ValueError:
            print('Fail to generate mesh for', i)
            # my_data.append([i, None])
            data_list.append(None)
    
    # obj_id = 0  # only one object
    # table_data.append([obj_id,image_list])
    table_data = []
    table_data.append(data_list)
    # log to wandb Table           
    # create a wandb.Table() with corresponding columns

    vis.destroy_window()
    return table_data, columns
