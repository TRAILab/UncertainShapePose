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

import wandb
import numpy as np


# pts: np.array([[x,y,z,color...]])
def wandb_vis_pointcloud_(name, pts_with_colors):
    wandb.log({f"pointcloud_{name}": wandb.Object3D(pts_with_colors)})

def wandb_vis_pointcloud(name, pts, colors):
    if colors is None:
        colors = np.zeros(pts.shape)
    scene_pcd_pts = np.concatenate([pts, colors], axis=-1)
    wandb_vis_pointcloud_(name, scene_pcd_pts)

def get_wandb_object_3d_from_points_color(pts, colors=None):
    if colors is None:
        colors = np.zeros(pts.shape)
    scene_pcd_pts = np.concatenate([pts, colors], axis=-1)
    return wandb.Object3D(scene_pcd_pts)

# load from file (obj)
def wandb_vis_mesh(name, file_dir):
    wandb.log({f'mesh_{name}' : wandb.Object3D(open(file_dir)) })

def wandb_vis_image(name, file_dir):
    wandb.log({f"image_{name}": wandb.Image(file_dir)})
