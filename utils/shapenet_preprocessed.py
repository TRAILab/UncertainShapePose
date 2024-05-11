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


from deep_sdf.data import get_instance_filenames, read_sdf_samples_into_ram
import deep_sdf.workspace as ws

import torch
import json
import os
import logging

from torch.utils.data import Dataset

from utils.io import sample_from_gt_mesh
import open3d as o3d

import numpy as np

# Utils function from: QUICK_LINKS/DeepSDF/reconstruct.py
def filter_data_sdf(data_sdf, method):
    # Attention: input data_sdf is a list with two positive, negative values.
    # We can cat them here.
    data_filter_list = []
    for data in data_sdf:
        if method == 'pos':
            # select all positive values
            data_filter = data[data[:,-1] > 0]
        elif method == 'neg':
            data_filter = data[data[:,-1] < 0]
        elif method == 'r0.1':
            data_filter = data[data[:,-1].abs() < 0.1]
        elif method == 'r0.1-0':
            # and we set it to 0
            data_filter = data[data[:,-1].abs() < 0.1]
            data_filter[:,-1] = 0
        elif method == 'full':
            # no filtering
            data_filter = data
        if len(data_filter) > 0:
            data_filter_list.append(data_filter)
    return torch.cat(data_filter_list, 0)


class ShapeNetPreprocessed(Dataset):
    def __init__(self, split_filename, data_source, dataset_source = 'deepsdf', 
        data_filter_method = 'full', # unused yet.
        random_filenames = False, 
        random_sdf = False):

        # load filenmae list from .json

        ### New info
        # view_file = './view_file_deepsdf.json'

          # deepsdf/pointcloud, sample from gt mesh, with sufficient postive and negative samples

        with open(split_filename, "r") as f:
            split = json.load(f)

        npz_filenames = get_instance_filenames(data_source, split)

        if random_filenames:
            random.shuffle(npz_filenames)

        self.npz_filenames = npz_filenames

        ## Init dirs
        self.data_source = data_source
        self.dataset_source = dataset_source
        self.data_filter_method = data_filter_method
        # if self.dataset_source == 'pointcloud':
        # TODO: change them
        self.groundtruth_meshes_dir = 'data/ShapeNetCore.v2/'
        self.normalization_params_dir = 'Thirdparty/DeepSDF/data/'

    def load_data(self, index):
        # Load mesh, normalization file name for data_source pointcloud.
        # they are also used for visualization.
        if not (index >= 0 and index < len(self.npz_filenames)):
            return None

        npz = self.npz_filenames[index]

        if self.dataset_source == 'deepsdf':
            full_filename = os.path.join(self.data_source, ws.sdf_samples_subdir, npz)
            logging.debug("loading {}".format(npz))
            data_sdf = read_sdf_samples_into_ram(full_filename)
            # a data filter
            data_sdf = filter_data_sdf(data_sdf, method = self.data_filter_method)
        elif self.dataset_source == 'pointcloud-vertex':
            # load data from pts.
            sample_pts_num = 2048
            data_sdf = self.load_gt_surface_sdfs(index, sample_pts_num, method='vertex')
        elif self.dataset_source == 'pointcloud':
            # load data from pts.
            sample_pts_num = 2048
            data_sdf = self.load_gt_surface_sdfs(index, sample_pts_num, method='pointcloud')
        else:
            raise NotImplementedError('please check dataset_source:', dataset_source)
        print('sdf samples num:', len(data_sdf))

        # randomnize?
        data_sdf = data_sdf[torch.randperm(data_sdf.shape[0])]

        return data_sdf

    def load_gt_surface_sdfs(self, index, sample_pts_num = 2048, method='pointcloud'):
        if not (index >= 0 and index < len(self.npz_filenames)):
            return None

        npz = self.npz_filenames[index]
        relative_name_str = os.path.join(*npz.split('/')[-2:])[:-4]
        gt_mesh_filename = os.path.join(self.groundtruth_meshes_dir, relative_name_str, 'models/model_normalized.obj')
        normalization_params_filename = os.path.join(
            self.normalization_params_dir,
            "NormalizationParameters",
            'ShapeNetV2',
            relative_name_str + ".npz",
        )
        
        # load data from pts.
        data_sdf = sample_from_gt_mesh(gt_mesh_filename, normalization_params_filename, sample_pts_num, method)
        data_sdf = torch.from_numpy(data_sdf).float()        

        return data_sdf
    

    def load_gt_mesh(self, index):
        if not (index >= 0 and index < len(self.npz_filenames)):
            return None

        npz = self.npz_filenames[index]
        relative_name_str = os.path.join(*npz.split('/')[-2:])[:-4]
        gt_mesh_filename = os.path.join(self.groundtruth_meshes_dir, relative_name_str, 'models/model_normalized.obj')
        normalization_params_filename = os.path.join(
            self.normalization_params_dir,
            "NormalizationParameters",
            'ShapeNetV2',
            relative_name_str + ".npz",
        )

        ply = o3d.io.read_triangle_mesh(gt_mesh_filename)
        normalization_params = np.load(normalization_params_filename)

        ply = ply.translate(normalization_params["offset"]).scale(normalization_params["scale"], center=(0,0,0))

        return ply

    def get_frame_name(self,index):
        if not (index >= 0 and index < len(self.npz_filenames)):
            return None

        npz = self.npz_filenames[index]

        return npz.split('/')[-1][:-4]

    def __len__(self):
        return len(self.npz_filenames)