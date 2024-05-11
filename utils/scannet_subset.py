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


# add dir
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.scannet import ScanNet
from utils.scannet import project_point_to_image, project_point_to_image_batch

from reconstruct.utils import ForceKeyErrorDict

import numpy as np

import os
import cv2

class ScanNetSubset(ScanNet):
    '''
    A bridge between ScanNet and the reconstruction demo.

    Init a scannet subset with specified scenes and object id.
    Then we can load frames from the observations of the specified object.
    '''

    def __init__(self, root_dir, scene_name, obj_id, configs=None, load_image=True, mask_path_root=None):
        '''
        @ configs: if you want to detect online
        '''
        super(ScanNetSubset, self).__init__(root_dir)

        self.scene_name = scene_name
        self.obj_id = obj_id

        # load observations
        self.observations = self.load_object_observations_from_scene(scene_name, obj_id, load_image=load_image)
        # object gt information is stored as:
        # 'scene_name': scene_name,
        # 'obj_id': obj_id,
        # 'object_mesh': object_mesh,
        # 'trs': trs,
        # 't_world_obj': t_world_obj,

        self.frames = self.observations['frames']

        self.index_transform = self.observations['index_transform']
        if self.index_transform is None:
            print('Warning: index_transform is None, we can not access to GT instance segmentation.')
        self.gt_instance_available = self.index_transform is not None

        self.detector = None

        self.configs = configs
        
        self.category = self.observations['category']

        self.mask_path_root = mask_path_root  # path to store instance_filt

    def get_frame_name(self, sub_id):
        '''
        @sub_id : not the final frame of the scene
        '''

        frame = self.frames[sub_id]
        frame_id = frame.frame_id

        return f'{self.scene_name}_f{frame_id}_ins{self.obj_id}_sid{sub_id}'

    def _sample_surface_points_from_depth(self, depth, K, mask):
        '''
        Sample only the depth inside a bounding box

        @depth: depth image (H, W)
        @K: calibration matrix, (3,3)
        @mask: (H, W), 1 for the object, 0 for the background

        @return: pointcloud (N, 3) in camera coordinate
        '''

        # resize depth to the same size as mask
        depth_large = cv2.resize(depth, mask.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # normalize unit from mm to m
        depth_large = depth_large / 1000.0

        # consider all the points in the depth image
        # unproject them into a point cloud
        depth_large[~mask] = 0

        # get bbox area from mask (u_min, v_min, u_max, v_max)
        valid_indices = np.where(mask)

        # if valid indices is zero
        if len(valid_indices[0]) == 0:
            raise ValueError('No valid indices in the mask.')

        bbox_area = np.array([np.min(valid_indices[1]), np.min(valid_indices[0]), np.max(valid_indices[1]), np.max(valid_indices[0])])

        # generate the pixel grid
        uu = np.arange(bbox_area[0], bbox_area[2])
        vv = np.arange(bbox_area[1], bbox_area[3])
        uu, vv = np.meshgrid(uu, vv)
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)

        # unproject the points
        # (u,v,1) -> (x,y,z)
        Kinv = np.linalg.inv(K)
        uv1 = np.stack([uu,vv,np.ones_like(uu)], axis=1)
        xyz = uv1 @ Kinv.T

        depth_mask_crop = depth_large[bbox_area[1]:bbox_area[3], bbox_area[0]:bbox_area[2]]

        xyz = xyz * depth_mask_crop.reshape(-1,1)

        # remove invalid points
        valid = depth_mask_crop.reshape(-1)>0
        xyz = xyz[valid]

        # transform to world coordinate
        # pts_world = t_world_cam[:3,:3] @ xyz.T + t_world_cam[:3,3:4]
        # pts_world = pts_world.T

        pts_cam = xyz
        return pts_cam

    # a version that generate the whole depth image
    # def _sample_surface_points_from_depth(self, depth, K, t_world_cam):
    #     '''
    #     @depth: depth image (H, W)
    #     @K: calibration matrix, (3,3)

    #     @return: pointcloud (N, 3) in camera coordinate
    #     '''
    #     # consider all the points in the depth image
    #     # unproject them into a point cloud

    #     # generate the pixel grid
    #     h, w = depth.shape
    #     u = np.arange(w)
    #     v = np.arange(h)
    #     uu, vv = np.meshgrid(u, v)
    #     uu = uu.reshape(-1)
    #     vv = vv.reshape(-1)

    #     # unproject the points
    #     # (u,v,1) -> (x,y,z)
    #     Kinv = np.linalg.inv(K)
    #     uv1 = np.stack([uu,vv,np.ones_like(uu)], axis=1)
    #     xyz = uv1 @ Kinv.T
    #     xyz = xyz * depth.reshape(-1,1)

    #     # remove invalid points
    #     valid = depth.reshape(-1)>0
    #     xyz = xyz[valid]

    #     return xyz


    def _generate_rays(self, depth, K, mask):       
        '''
        the lidar points are depth observations
        So the rays are the projected lidar points;
        For RGB-D case, the rays are infact all the pixels inside the bbox 
            1) Foreground area: have valid depth (some are the surface, some are not, but with large values)
            2) Background area: no depths, Unkown area
        
        We only consider points that have valid depth, and inside the bbox.

        @ return: rays (N1+N2, 3) in camera coordinate; N1: foreground points, N2: background points
                  depths (N1, 1)
        '''

        # All pixels inside mask bbox
        # get bbox area from mask (u_min, v_min, u_max, v_max)
        valid_indices = np.where(mask)  #(Height (y),Width (x))
        bbox_area = np.array([np.min(valid_indices[1]), np.min(valid_indices[0]), np.max(valid_indices[1]), np.max(valid_indices[0])])

        # generate the pixel grid
        # uu = np.arange(bbox_area[0], bbox_area[2])  # cover the bbox (N_x, )
        # vv = np.arange(bbox_area[1], bbox_area[3])  # (N_y, )
        # uu_grid, vv_grid = np.meshgrid(uu, vv)  # (N_y, N_x)
        # uu_vec = uu_grid.reshape(-1)
        # vv_vec = vv_grid.reshape(-1)

        # Keep those have valid depth values
        depth_large = cv2.resize(depth, mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
        depth_large = depth_large / 1000.0

        # foreground: depth inside mask w/ valid depth
        # depth inside bbox
        # depth_bbox_crop = depth_large[bbox_area[1]:bbox_area[3], bbox_area[0]:bbox_area[2]]
        # depth inside mask of the bbox
        depth_bbox_crop_mask = depth_large[mask]     
        # depth obs (w/ valid depth)
        depth_bbox_crop_mask_valid = depth_bbox_crop_mask[depth_bbox_crop_mask>0]
        # reshape to vector
        depth_obs_foreground = depth_bbox_crop_mask_valid.reshape(-1)
        # get the rays for those valid depth values in foreground
        # get a mask from bbox to mask with valid depth
        mask_and_valid_depth = mask * (depth_large>0)
        # get indices of those True
        valid_indices = np.where(mask_and_valid_depth)
        # concat u,v of valid_indices
        uu_mask = valid_indices[1]
        vv_mask = valid_indices[0]
        
        # unproject to rays
        Kinv = np.linalg.inv(K)
        uv1 = np.stack([uu_mask,vv_mask,np.ones_like(uu_mask)], axis=1)
        rays = uv1 @ Kinv.T


        # Further sample points outside masks. (not all of them)
        mask_inside_bbox = np.full_like(mask, False)
        mask_inside_bbox[bbox_area[1]:bbox_area[3], bbox_area[0]:bbox_area[2]] = True
        mask_inside_bbox_not_mask = (mask_inside_bbox) * (~mask)
        valid_indices = np.where(mask_inside_bbox_not_mask)
        uu_mask = valid_indices[1]
        vv_mask = valid_indices[0]

        # unproject to rays
        uv2 = np.stack([uu_mask,vv_mask,np.ones_like(uu_mask)], axis=1)
        rays2 = uv2 @ Kinv.T

        # concat rays
        rays_output = np.concatenate([rays, rays2], axis=0)

        # keep the valid depth values
        depth_valid = depth_obs_foreground

        return rays_output, depth_valid



    def _associate_instance_id_scan2cad_to_scannet(self, mask_full, frame, t_world_obj):
        '''
        @mask_full: (H, W)
        @K: (3,3)
        @t_world_cam: (4,4)
        @t_world_obj: (4,4)

        @return: selected_id
        -   special case: the projected center is not in the image plane, return None
        '''

        K = frame.K
        t_world_cam = frame.t_world_cam

        uv = project_point_to_image(t_world_obj[:3,3], t_world_cam, K)
        u = int(uv[0])
        v = int(uv[1])

        # check if in image plane
        if u<0 or u>=mask_full.shape[1] or v<0 or v>=mask_full.shape[0]:
            selected_id = None
        else:
            selected_id = mask_full[v,u]

        debug=False
        if debug:
            # visualize mask and the projected center
            mask_vis = mask_full.copy()

            if selected_id is not None:
                mask_vis = cv2.circle(mask_vis, (u,v), 5, 255, -1)

            from skimage import color
            cv2.imwrite(f'output/mask_vis_nofilt_ins_proj_{frame.frame_id}.png', color.label2rgb(mask_vis)*255)

        return selected_id

    def _associate_instance_id_scan2cad_to_scannet_vote(self, mask_full, frame, observations):
        '''
        Project dense annotations into images and vote instance label
        '''

        keypoints_cad = observations['keypoints_cad']['position']
        keypoints_scan = observations['keypoints_scan']['position']
        # reshape a list x1,y1,z1,x2,y2,z2,... to (N, 3)
        keypoints_cad = np.array(keypoints_cad).reshape(-1,3)
        keypoints_scan = np.array(keypoints_scan).reshape(-1,3)

        # project all the points into images
        K = frame.K
        t_world_cam = frame.t_world_cam

        # uv: (8, 2)
        uv_batch = project_point_to_image_batch(keypoints_scan, t_world_cam, K)
        # u = int(uv[0])
        # v = int(uv[1])

        # vote
        id_list = []
        for uv in uv_batch:
            u = int(uv[0])
            v = int(uv[1])

            # check if in image plane
            if u<0 or u>=mask_full.shape[1] or v<0 or v>=mask_full.shape[0]:
                selected_id = None
            else:
                selected_id = mask_full[v,u]

            id_list.append(selected_id)
        
        # ignore None
        id_list_valid = [id for id in id_list if id is not None]

        if len(id_list_valid) == 0:
            # fail to get the instance id with 2d projection
            raise ValueError('Fail to get the instance id with 2d projection.')

        final_id = np.argmax(np.bincount(id_list_valid))

        debug=False
        if debug:
            # visualize mask and the projected center
            mask_vis = mask_full.copy()

            # if selected_id is not None:
                # mask_vis = cv2.circle(mask_vis, (u,v), 5, 255, -1)
            
            for uv in uv_batch:
                u = int(uv[0])
                v = int(uv[1])
                mask_vis = cv2.circle(mask_vis, (u,v), 5, 255, -1)

            from skimage import color
            cv2.imwrite(f'output/mask_vis_nofilt_ins_proj_{frame.frame_id}_batch.png', color.label2rgb(mask_vis)*255)

            # output selected mask
            mask_vis = mask_full.copy()
            mask_vis = mask_vis == final_id
            cv2.imwrite(f'output/mask_vis_nofilt_ins_proj_{frame.frame_id}_final.png', color.label2rgb(mask_vis)*255)

        return final_id


    def _load_mask(self, scene_name, obj_id, frame, use_gt=True,
                    gt_match_method = '3d_match'):
        '''
        @rgb: rgb image (H, W, 3)
        @return: mask (H, W)

        @use_gt: load from ScanNet annotation
        '''

        frame_id = frame.frame_id

        if use_gt:
            if not self.gt_instance_available:
                raise ValueError('GT instance segmentation is not available.')

            # load from gt dirs
            if self.mask_path_root is None:
                mask_path = os.path.join(self.root_dir, 'data', 'scans', scene_name, 'instance-filt')
            else:
                mask_path = os.path.join(self.mask_path_root, scene_name, 'instance-filt')
            # TODO: align with new sampler...

            scannet_frame_id = self.index_transform[frame_id]

            mask_path_image = os.path.join(mask_path, f'{scannet_frame_id}.png')

            # Check if exist
            if not os.path.exists(mask_path_image):
                AUTOMATIC_UNZIP = True
                if AUTOMATIC_UNZIP:
                    # unzip
                    import zipfile
                    # scene_name + 2d-instance-filt.zip
                    # scene_scan_data_dir = os.path.dirname(os.path.dirname(mask_path))
                    scene_scan_data_dir = os.path.join(self.root_dir, 'data', 'scans', scene_name)
                    
                    zip_file_name = scene_scan_data_dir + f'/{scene_name}_2d-instance-filt.zip'
                    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
                        mask_path_unzip = os.path.dirname(mask_path)
                        zip_ref.extractall(mask_path_unzip)
                    
                    print('Firt run. Zip instace-filt file to:', mask_path_unzip)

            mask_full = cv2.imread(mask_path_image, cv2.IMREAD_GRAYSCALE)

            t_world_obj = self.observations['t_world_obj']

            '''
            Legacy method: Project the center of Annotated 3D Pose to images, and find the mask area.
            '''
            if gt_match_method == 'project':
                '''
                Project center points; Have errors.
                '''
                selected_id = self._associate_instance_id_scan2cad_to_scannet(mask_full, frame, t_world_obj)
            elif gt_match_method == 'project_dense':
                '''
                Project all the annotated points into images and vote
                '''
                selected_id = self._associate_instance_id_scan2cad_to_scannet_vote(mask_full, frame, self.observations)
            elif gt_match_method == '3d_match':
                '''
                Use 3D matching to find the instance id
                '''
                # load gt_match_file
                # ind_file = os.path.join(self.scan2cad.root_dir, 'indices_to_scannet', scene_name+'_ind_2_scannet.npy')
                # ind_2_scannet = np.load(ind_file)

                ind_2_scannet = self.scan2cad.load_ind_2_scannet(scene_name)

                # get the index of the object
                ind_scannet = ind_2_scannet[obj_id]
                
                '''
                Add one to get instance label in instance segmentation map!
                '''

                # check if it is valid:  not -1
                if ind_scannet == -1:
                    print('Warning: ind_scannet == -1, no corresponding instance in Scan2CAD.')

                ind_scannet += 1

                # get the instance id
                selected_id = ind_scannet

            else:
                '''
                [NOT CORRECT]
                The Scan2CAD annotation has the same id as the instance id of the Scan2CAD
                '''
                selected_id = obj_id

            if selected_id is None:
                # fail to get the instance id with 2d projection
                raise ValueError('Fail to get the instance id with 2d projection.')

            # select the mask corresponding to current instance
            mask = mask_full == selected_id

        else:
            # load from online detector, e.g., Mask-RCNN
            # TODO: Finish this part
            raise NotImplementedError

            self.mask_save_dir = os.path.join(self.root_dir, 'data', 'masks')

            # off line detection, directly load from saved datas
            mask_path = os.path.join(self.mask_save_dir, scene_name, f'frame-{frame_id}', f'ins{obj_id}.png')

            # online detection, if not found, detect with Mask-RCNN, then select using projected center.
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                return mask
            
            # detect online
            resulst = self._detect_mask_online(frame.rgb)

            # projected object center
            # select the mask belonging to the GT object

            # TODO: how to get the GT Mask? Instance segmentation of 2D image?
            mask = select_mask_with_center(result, frame.K, frame.t_world_cam, frame.object_mesh)

            # select mask with the center

            # save mask
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, mask)

        return mask

    def _load_detector2d(self, method='maskrcnn'):
        from mmdet.apis import init_detector, inference_detector

        if method == 'maskrcnn':
            # config_file = 'configs/mask_rcnn/mask_rcnn_x101_32x8d_fpn_1x_coco.py'
            config_file = 'configs/mask_rcnn/mask-rcnn_x101-32x8d_fpn_1x_coco.py'
            checkpoint_file = 'models/mask_rcnn_x101_32x8d_fpn_1x_coco_20220630_173841-0aaf329e.pth'
        elif method == 'mask2former':
            config_file = 'configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
            checkpoint_file = 'models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'
        model = init_detector(config_file, checkpoint_file, device='cuda:0')

        return model

    def _detect_mask_online(self, rgb):
        '''
        @rgb: rgb image (H, W, 3)
        @return: mask (H, W)
        '''
        # Get mask from Mask-RCNN
        # if not initialized, load model
        if self.detector is None:
            # load it
            if self.configs is None:
                raise ValueError('Please specify the config file for detector2d.')

            # from reconstruct.detector2d import get_detector2d
            self.detector = self._load_detector2d()
        
        # detect
        result = inference_detector(self.detector, rgb)
        
        # mask = self.detector.detect(rgb)
        bboxes, masks, labels = result.pred_instances.bboxes, result.pred_instances.masks, result.pred_instances.labels

        debug_vis = False
        if debug_vis:
            im_vis = rgb.copy()
            for id in range(len(bboxes)):
                # draw bbox, mask, label
                bbox = bboxes[id].detach().cpu().numpy().astype(np.int32)
                mask = masks[id].detach().cpu().numpy()
                label = labels[id].detach().cpu().numpy()

                # draw bbox
                im_vis = cv2.rectangle(im_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)

                # draw mask
                # im_vis = self.detector.draw_mask(im_vis, mask, (255,0,0))
                # im_vis = cv2.addWeighted(im_vis, 0.5, mask, 0.5, 0)

                # draw label
                # im_vis = self.detector.draw_label(im_vis, label, bbox, (255,0,0))
                # change label id to text with coco
                # label_coco = self.detector.coco.cats[label]['name']
                im_vis = cv2.putText(im_vis, str(label), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.imwrite('vis.png', im_vis)

        resulst = {
            'bboxs': bboxes,
            'masks': masks,
            'labels': labels
        }

        # visualzie result into a image
        # vis = self.detector.visualize(rgb, result)
        # cv2.imshow('vis', vis)
        # cv2.waitKey(0)


        return resulst

    def get_t_obj_deepsdf(self):
        '''
        An unfinished function, to get the transformation of object frame (normalized obj)
        to DeepSDF (output of the shape model)

        There is a normalized transformation to align all the models into a 1-scale space.
        The normalized files are different for each instances, and are stored as .npz files during
        the training of the DeepSDF model.

        Now we use an approximate and same transform to initialize the objects.
        '''

        # The transformation is a SIM(3) with scale of 2
        t_obj_deepsdf = np.eye(4, dtype=np.float32)
        t_obj_deepsdf[0:3, 0:3] = np.eye(3, dtype=np.float32) / 2.0

        return t_obj_deepsdf

    def transform_obj_to_deepsdf_gt(self, T_bo_origin,
                                    preprocessed_dir='data/shapenet_preprocessed'):
        '''
        @ input: T_bo_origin
        
        @ return: T_bo_norm_gt
        '''
        norm_file_name = os.path.join(self.root_dir, preprocessed_dir, 
                                'NormalizationParameters', 
                                self.observations['catid_cad'], 
                                self.observations['id_cad'] + '.npz')
        normalization_params = np.load(norm_file_name)
        offset_no = normalization_params["offset"]
        scale_no = normalization_params["scale"]
        
        T_bo_norm = T_bo_origin.copy()
        T_bo_norm[:3,:3] = T_bo_norm[:3,:3] @ (np.eye(3) / scale_no)
        T_bo_norm[:3,3] = T_bo_norm[:3,3] - offset_no

        
        return T_bo_norm

    def get_frame_by_id(self, sub_id):
        '''
        Output structure:
        a list of [det, ...]

            det = ForceKeyErrorDict()
            det.T_cam_obj = np.eye(4,4,dtype=np.single)
            det.surface_points = inputs.cpu().numpy()
            det.rays = None
            det.depth = None
            det.gt_local = gt.cpu().numpy()
            det.label = label

        '''
        
        frame = self.frames[sub_id]

        # if frame has not been initialized images, load now
        if frame.rgb is None:
            frame = self.load_frame(self.scene_name, frame.frame_id, load_image=True)

        num_ins = 1

        # load object info
        t_world_obj = self.observations['t_world_obj']

        # load information into a new dict
        t_world_cam = frame.t_world_cam
        t_cam_world = np.linalg.inv(t_world_cam)
        T_cam_obj = t_cam_world @ t_world_obj

        # TODO: preprocess RGB/Depth Image with Object Masks.
        # TODO: generate point cloud from masked images.
        # load an object detector! and masks!

        det = ForceKeyErrorDict()
        det.T_cam_obj = T_cam_obj.astype(np.float32)
        det.surface_points = None
        det.rays = None
        det.depth = None
        det.gt_local = None
        det.label = None

        # those are needed for reconstruction:
        # det.surface_points,  ## used for 3d loss;
        # det.rays, ## render 2d images
        # det.depth ## depth image

        # step 1: load RGB image and generate masks
        # input: scene_name, instance_id, rgb
        mask = self._load_mask(self.scene_name, self.obj_id, frame)

        # visualize mask
        # from skimage import color
        # cv2.imwrite('output/mask_vis_nofilt_ins.png', color.label2rgb(mask)*255)

        # sparsely sample surface points from depth image
        det.surface_points = self._sample_surface_points_from_depth(frame.depth, frame.K, mask)
        # use float 32
        det.surface_points = det.surface_points.astype(np.float32)

        # add surface_points_world
        det.surface_points_world = (t_world_cam[:3,:3] @ det.surface_points.T + t_world_cam[:3,3:4]).T


        # generate rays from surface points
        ## Consider instance segmentation mask
        ## - Foreground area / background area
        ## Sample points from both areas
        det.rays, det.depth = self._generate_rays(frame.depth, frame.K, mask)
        # use float 32
        det.rays = det.rays.astype(np.float32)
        det.depth = det.depth.astype(np.float32)

        '''
        Extract depth observations from depth map.
        Each ray corresponds to a depth value, with the same order.
        '''
        # det.depth = frame.depth

        '''
        Temp transform update with obj to DeepSDF
        '''
        t_obj_deepsdf_coarse = self.get_t_obj_deepsdf()
        det.t_obj_deepsdf_coarse = t_obj_deepsdf_coarse
        det.t_cam_deepsdf_coarse = det.T_cam_obj @ t_obj_deepsdf_coarse
        
        det.T_world_cam = t_world_cam.astype(np.float32)
        
        det.t_world_obj = t_world_obj

        # debug = False
        # if debug:
        #     t_world_cam = frame.t_world_cam
            
        #     # add 1 to final dim of det.surface_points
        #     pts_cam = det.surface_points
        #     pts_world = t_world_cam[:3,:3] @ pts_cam.T + t_world_cam[:3,3:4]
        #     pts_world = pts_world.T

        #     # save pts in ply as o3d
        #     # import open3d as o3d
        #     # pts_o3d = o3d.geometry.PointCloud()
        #     # pts_o3d.points = o3d.utility.Vector3dVector(pts_world)

        #     # # save the point cloud
        #     # o3d.io.write_point_cloud(f'./output/pts_from_depth_world_m3.ply', pts_o3d)

        #     # save valid depth map
        #     depth_valid = frame.depth.copy()
        #     # resize to the same size as mask
        #     depth_valid = cv2.resize(depth_valid, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        #     depth_valid[~mask] = 0

        #     depth_valid_vis = depth_valid / depth_valid.max() * 255
        #     # save
        #     cv2.imwrite(f'./output/pts_from_depth_world_m3_depth.png', depth_valid_vis)

        #     # new test: transform pts from camera frame to object frame
        #     t_obj_cam = np.linalg.inv(T_cam_obj)
        #     pts_obj = t_obj_cam[:3,:3] @ pts_cam.T + t_obj_cam[:3,3:4]
        #     pts_obj = pts_obj.T

        #     # save pts in ply as o3d
        #     pts_o3d_obj = o3d.geometry.PointCloud()
        #     pts_o3d_obj.points = o3d.utility.Vector3dVector(pts_obj)

        #     # save the point cloud
        #     o3d.io.write_point_cloud(f'./output/pts_from_depth_obj_m3.ply', pts_o3d_obj)

        #     # save a model in object frame
        #     obj_mesh = self.observations['object_mesh']
        #     # save
        #     o3d.io.write_triangle_mesh(f'./output/mesh_obj.ply', obj_mesh)
            

        det_list = [det]
        return det_list

    def get_one_frame(self, sub_id, load_image=True):
        if sub_id >= len(self):
            raise IndexError(f'Index {sub_id} out of range {len(self)}')
        frame = self.frames[sub_id]

        # if frame rgb images are not loaded, load here
        if load_image and frame.rgb is None:
            frame = self.load_frame(self.scene_name, frame.frame_id, load_image=True)
        
        return frame


    def __len__(self) -> int:
        return len(self.frames)

    def get_gt_bbox_world(self, gt_mesh, gt_t_world_obj, n_sample_pts=10000, device = 'cuda'):
        '''
        @ return: bbox1_world (8, 3)
        '''
        # sample points in origin mesh coordinate
        pts1_sampled = gt_mesh.sample_points_uniformly(number_of_points=n_sample_pts)

        import torch
        pts1_sampled = torch.from_numpy(np.asarray(pts1_sampled.points)).float().unsqueeze(0).to(device)

        # fit bbox
        from utils.evo import fit_cuboid_to_points
        bbox1 = fit_cuboid_to_points(pts1_sampled.squeeze(0))

        gt_t_world_obj = torch.from_numpy(gt_t_world_obj).float().to(device)

        # bbox to world
        bbox1_world = bbox1 @ gt_t_world_obj[:3,:3].T + gt_t_world_obj[:3,3]

        return bbox1_world

    def get_observation_ratio(self, sub_id):
        '''
        Get the ratio of valid observations in a frame

        @sub_id: order of all the frames, not the real frame_id.
        '''
        frame = self.frames[sub_id]
        mask = self._load_mask(self.scene_name, self.obj_id, frame)

        gt_t_world_cam = frame.t_world_cam

        from utils.evo import calculate_ob_ratio_from_projection

        gt_mesh = self.observations['object_mesh']
        # if not loaded, load it
        if isinstance(gt_mesh, str):
            gt_mesh = self.scan2cad.load_shape(self.observations['catid_cad'], self.observations['id_cad'])
            # gt_mesh = self.scan2cad.load_shape(obj['catid_cad'], obj['id_cad'])
            self.observations['object_mesh'] = gt_mesh

        gt_t_world_obj = self.observations['t_world_obj']
        if not 'gt_bbox_world' in self.observations:
            gt_bbox_world = self.get_gt_bbox_world(gt_mesh, gt_t_world_obj).cpu().numpy()
            self.observations['gt_bbox_world'] = gt_bbox_world
        else:
            gt_bbox_world = self.observations['gt_bbox_world']

        # check if gt_t_world_cam is valid
        if np.isnan(gt_t_world_cam).any():
            # invalid
            ob_ratio = 0
        else:
            ob_ratio = calculate_ob_ratio_from_projection(gt_bbox_world, gt_t_world_cam, mask, frame.K)

        return ob_ratio

    def get_gt_mesh(self):
        '''
        Get the ground truth mesh
        '''
        gt_mesh = self.observations['object_mesh']
        # if not loaded, load it
        if isinstance(gt_mesh, str):
            gt_mesh = self.scan2cad.load_shape(self.observations['catid_cad'], self.observations['id_cad'])
            # gt_mesh = self.scan2cad.load_shape(obj['catid_cad'], obj['id_cad'])
            self.observations['object_mesh'] = gt_mesh

        return gt_mesh
    
    def get_gt_obj_pose_t_world_obj(self):
        '''
        Get the ground truth object pose
        '''
        gt_t_world_obj = self.observations['t_world_obj']
        return gt_t_world_obj

    def get_gt_obj_pose_t_world_obj_norm(self, preprocessed_dir='data/shapenet_preprocessed'):
        '''
        Further load the normalization file.
        '''

        norm_file_name = os.path.join(self.root_dir, preprocessed_dir, 
                                      'NormalizationParameters', 
                                      self.observations['catid_cad'], 
                                      self.observations['id_cad'] + '.npz')
        normalization_params = np.load(norm_file_name)
        offset_no = normalization_params["offset"]
        scale_no = normalization_params["scale"]

        gt_t_world_obj = self.observations['t_world_obj']
        gt_t_world_obj_norm = gt_t_world_obj.copy()
        gt_t_world_obj_norm[:3,3] = gt_t_world_obj_norm[:3,3] / scale_no - offset_no

        return gt_t_world_obj_norm


    def get_gt_sampled_points_in_world(self, n_sample_pts=10000):
        import torch 

        gt_mesh = self.get_gt_mesh()

        pts1_sampled = gt_mesh.sample_points_uniformly(number_of_points=n_sample_pts)
        pts1_sampled = torch.from_numpy(np.asarray(pts1_sampled.points)).float().unsqueeze(0)

        # to gpu
        gt_t_world_obj = self.get_gt_obj_pose_t_world_obj()
        gt_t_world_obj = torch.from_numpy(gt_t_world_obj).float()

        # transform points, and bbox to world (N, 3) @ (4,4) -> (N, 3)
        pts1_sampled_world = pts1_sampled.squeeze(0) @ gt_t_world_obj[:3,:3].T + gt_t_world_obj[:3,3]

        return pts1_sampled_world


'''
Test FUNCTIONS
'''

    
def test_gt_mask_association():
    '''
    Output contiuous frames to check the segmented masks
    '''
    # obj_id = 9 # 9,13 window
    # obj_id = 5 # 5 is the floor
    obj_id = 9
    dataset = ScanNetSubset(root_dir='data/scannet', scene_name='scene0568_00', obj_id=obj_id)

    # load frames
    frames = dataset.frames

    num_max_frame = len(frames)
    # randomly select 10 frames
    frames_selected = np.random.choice(num_max_frame, 10, replace=False)

    # save mask
    mask_save_dir = 'output/debug/mask_gt' + f'/obj-{obj_id}-3dmatch'
    os.makedirs(mask_save_dir, exist_ok=True)

    # frames_selected = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    frames = [frames[i] for i in frames_selected]
    for frame in frames:
        mask = dataset._load_mask(dataset.scene_name, dataset.obj_id, frame, gt_match_method='3d_match')
        rgb = frame.rgb
        # bgr 2 rgb
        rgb = rgb[:,:,::-1]

        # concat rgb and mask; note mask is a bool type, change it to 255
        mask_vis = mask.copy()
        mask_vis = (mask_vis * 255).astype(np.uint8)
        mask_vis = np.stack([mask_vis, mask_vis, mask_vis], axis=2)
        mask_vis = np.concatenate([rgb, mask_vis], axis=1)

        # save
        mask_save_name = os.path.join(mask_save_dir, f'{frame.frame_id}.png')
        import matplotlib.pyplot as plt
        plt.imsave(mask_save_name, mask_vis)

    print('Done.')





if __name__ == '__main__':

    test_gt_mask_association()