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


import argparse

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_kitti.json', help='path to config file')
    parser.add_argument('-d', '--sequence_dir', type=str, default='data/KITTI/07', required=False, help='path to kitti sequence')
    parser.add_argument('-i', '--frame_id', type=int, required=False, default=None, help='frame id')
    parser.add_argument('-s', '--split_filename', type=str, required=False, help='.json split files for ShapeNetPreprocessed.')
    parser.add_argument('--data_source', type=str, required=False, help='data set path for ShapeNetPreprocessed.')
    parser.add_argument('--dataset_source', type=str, required=False, help='choose type: pointcloud, deepsdf sampling etc.', default='deepsdf')
    parser.add_argument('--loss_type', type=str, default='energy_score')
    parser.add_argument('--mask_method', type=str, required=False, default=None)
    parser.add_argument('--num_iterations', type=int, required=False, default=200)
    parser.add_argument('--jump', type=bool, required=False, default=False)
    parser.add_argument('--cur_time', type=str, required=False, default=None)
    parser.add_argument('--init_sigma', type=float, required=False, default=0.001, help="when using uncertainty reconstruction, define the sigma of initialized codes.")
    parser.add_argument('--prefix', type=str, required=False, default='UncertainMapping', help="The name of this exp.")
    parser.add_argument('--random_seed', type=int, required=False, default=1, help="The random seed of the whole script.")
    parser.add_argument('--use_2d_loss', required=False, default=False, action='store_true', help="Use 2d loss.")

    # add sample_num
    parser.add_argument('--sample_num', type=int, required=False, default=10, help="The number of sample for uncertainty.")

    # # add visualize_intermediate
    # parser.add_argument('--visualize_intermediate', required=False, default=False, action='store_true', help="Visualize intermediate results.")

    # add visualize_intermediate
    parser.add_argument('--visualize_intermediate', required=False, default=True, action='store_true', help="Visualize intermediate results.")
    parser.add_argument('--close_visualize_intermediate', dest='visualize_intermediate', action='store_false', help="Close visualization of intermediate results.")

    # # add bool vis_abs_uncer
    # parser.add_argument('--vis_abs_uncer', required=False, default=False, action='store_true', help="Visualize absolute uncertainty.")
    # add vis_abs_uncer
    parser.add_argument('--vis_abs_uncer', required=False, default=True, action='store_true', help="Visualize absolute uncertainty.")
    parser.add_argument('--no_vis_abs_uncer', dest='vis_abs_uncer', action='store_false', help="Don't visualize absolute uncertainty.")

    # add for scannet dataset: args.scene_name, args.obj_id
    parser.add_argument('--scene_name', type=str, required=False, default=None, help="The scene name of scannet dataset.")
    parser.add_argument('--obj_id', type=int, required=False, default=None, help="The object id of scannet dataset.")

    # dataset_name
    parser.add_argument('--dataset_name', type=str, required=False, default='scannet', help="The dataset name.")

    # save_root
    parser.add_argument('--save_root', type=str, required=False, default='./output', help="The root dir to save results.")

    # add loss_type_2d_uncertain
    parser.add_argument('--loss_type_2d_uncertain', type=str, required=False, default='energy_score', help="The loss type for 2d loss.")

    # # whether to use gt_association
    # parser.add_argument('--use_gt_association', required=False, default=False, action='store_true', help="Use gt association.")
    # add use_gt_association
    parser.add_argument('--use_gt_association', required=False, default=True, action='store_true', help="Use gt association.")
    parser.add_argument('--no_gt_association', dest='use_gt_association', action='store_false', help="Don't use gt association.")

    # add --dataset_subset_package, continue_from_scene
    parser.add_argument('--dataset_subset_package', type=str, required=False, default=None, help="The dataset subset package.")
    parser.add_argument('--continue_from_scene', type=str, required=False, default=None, help="The scene name to continue from.")

    # consider view num; if 1, then single-view; if more than 1, multi-view
    parser.add_argument('--view_num', type=int, required=False, default=10, help="The number of views to consider.")

    # a param to change init method, two options: gt_noise, estimator
    parser.add_argument('--pose_init_method', type=str, required=False, default='estimator', help="The init method for pose.")

    # add debug option
    parser.add_argument('--debug', required=False, default=False, action='store_true', help="Debug mode.")

    # add close_2d_loss, close_3d_loss
    parser.add_argument('--close_2d_loss', required=False, default=False, action='store_true', help="Close 2d loss.")
    parser.add_argument('--close_3d_loss', required=False, default=False, action='store_true', help="Close 3d loss.")

    # add lr
    parser.add_argument('--lr', type=float, required=False, default=0.005, help="The learning rate.")

    # add init_sigma_pose, init_sigma_scale
    parser.add_argument('--init_sigma_pose', type=float, required=False, default=0.01, help="The init sigma for pose.")
    parser.add_argument('--init_sigma_scale', type=float, required=False, default=0.01, help="The init sigma for scale.")

    # add weight_3d
    parser.add_argument('--weight_3d', type=float, required=False, default=100, help="The weight for 3d loss.")
    # add weight_2d
    parser.add_argument('--weight_2d', type=float, required=False, default=50, help="The weight for 2d loss.")
    # add weight_norm
    parser.add_argument('--weight_norm', type=float, required=False, default=3000, help="The weight for norm loss.")

    # # add open_visualization
    # parser.add_argument('--open_visualization', required=False, default=False, action='store_true', help="Open visualization.")

    # add open_visualization
    parser.add_argument('--open_visualization', required=False, default=True, action='store_true', help="Open visualization.")
    parser.add_argument('--close_visualization', dest='open_visualization', action='store_false', help="Close visualization.")

    # add render_2d_K, default 400
    parser.add_argument('--render_2d_K', type=int, required=False, default=400, help="The K for 2d rendering.")

    # add render_2d_calibrate_C, default 1.0
    parser.add_argument('--render_2d_calibrate_C', type=float, required=False, default=1.0, help="The C for 2d rendering.")
    
    # add render_2d_const_a
    parser.add_argument('--render_2d_const_a', type=float, required=False, default=0.0, help="The const a for 2d rendering.")

    # add option_select_frame
    parser.add_argument('--option_select_frame', type=str, required=False, default='close')
    
    # add mask_path_root, default None
    parser.add_argument('--mask_path_root', type=str, required=False, default=None)
    
    # add noise_level, default 1.0
    parser.add_argument('--noise_level', type=float, required=False, default=1.0, help="Noise level for gt_noise pose initialization method.")

    # add close_pose_optimization 
    parser.add_argument('--close_pose_optimization', required=False, default=False, action='store_true', help="Close pose optimization gradients.")
    
    # add view_file
    parser.add_argument('--view_file', type=str, required=False, default='./view_param.json')

    return parser
