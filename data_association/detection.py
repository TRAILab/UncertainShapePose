from addict import Dict

import cv2
import numpy as np



class ForceKeyErrorDict(Dict):
    def __missing__(self, key):
        # raise KeyError(key)
        print('miss key: ', key)

class Detection():
    def __init__(self, object, obs_idx):
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

        # load information into a new dict
        rgb = object.observations[obs_idx].rgb
        depth = object.observations[obs_idx].depth
        mask = object.observations[obs_idx].mask_inflated

        T_cam_obj = np.linalg.inv(object.observations[obs_idx].T) @ object.estimated_pose

        K = object.observations[obs_idx].K

        self.det = ForceKeyErrorDict()
        self.det.T_world_cam = object.observations[obs_idx].T.astype(np.float32)
        self.det.T_cam_object = T_cam_obj
        self.det.T_cam_deepsdf_coarse = None
        self.det.T_obj_deepsdf_coarse = None
        self.det.surface_points = None
        self.det.surface_points_world = None
        self.det.num_surface_points = None
        self.det.rays = None
        self.det.depths = None
        self.det.gt_local = None
        self.det.label = None

        # those are needed for reconstruction:
        # det.surface_points,  ## used for 3d loss;
        # det.rays, ## render 2d images
        # det.depth ## depth image

        # sparsely sample surface points from depth image
        self.det.surface_points = self._sample_surface_points_from_depth(depth, K, mask)
        self.det.num_surface_points = self.det.surface_points.shape[0]
        self.det.surface_points_world = (np.hstack((self.det.surface_points, np.ones((self.det.num_surface_points, 1)))) @ self.det.T_world_cam.T )[:,0:3]
        
        
        # generate rays from surface points
        ## Consider instance segmentation mask
        ## - Foreground area / background area
        ## Sample points from both areas
        self.det.rays, self.det.depths = self._generate_rays(depth, K, mask)

        '''
        Temp transform update with obj to DeepSDF
        '''
        self.det.T_obj_deepsdf_coarse = self.get_t_obj_deepsdf(object)
        self.det.T_cam_deepsdf_coarse = (self.det.T_cam_object @ self.det.T_obj_deepsdf_coarse).astype(np.float32)

    def _sample_surface_points_from_depth(self, depth, K, mask):
        '''
        Sample only the depth inside a bounding box

        @depth: depth image (H, W)
        @K: calibration matrix, (3,3)
        @mask: (H, W), 1 for the object, 0 for the background

        @return: pointcloud (N, 3) in camera coordinate
        '''

        # resize depth to the same size as mask
        

        # normalize unit from mm to m
        depth_metric = depth / 1000.0

        mask_bool = mask != 0

        # consider all the points in the depth image
        # unproject them into a point cloud
        depth_metric[~mask_bool] = 0

        # get bbox area from mask (u_min, v_min, u_max, v_max)
        valid_indices = np.where(mask_bool)
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

        depth_mask_crop = depth_metric[bbox_area[1]:bbox_area[3], bbox_area[0]:bbox_area[2]]

        xyz = xyz * depth_mask_crop.reshape(-1,1)

        # remove invalid points
        valid = depth_mask_crop.reshape(-1)>0
        xyz = xyz[valid]

        # transform to world coordinate
        # pts_world = t_world_cam[:3,:3] @ xyz.T + t_world_cam[:3,3:4]
        # pts_world = pts_world.T

        pts_cam = xyz.astype(np.float32)
        return pts_cam
    

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
        mask_bool = mask != 0
        valid_indices = np.where(mask_bool)  #(Height (y),Width (x))
        bbox_area = np.array([np.min(valid_indices[1]), np.min(valid_indices[0]), np.max(valid_indices[1]), np.max(valid_indices[0])])
        # generate the pixel grid
        # uu = np.arange(bbox_area[0], bbox_area[2])  # cover the bbox (N_x, )
        # vv = np.arange(bbox_area[1], bbox_area[3])  # (N_y, )
        # uu_grid, vv_grid = np.meshgrid(uu, vv)  # (N_y, N_x)
        # uu_vec = uu_grid.reshape(-1)
        # vv_vec = vv_grid.reshape(-1)

        # Keep those have valid depth values
        depth_metric = depth / 1000.0

        # foreground: depth inside mask w/ valid depth
        # depth inside bbox
        # depth_bbox_crop = depth_large[bbox_area[1]:bbox_area[3], bbox_area[0]:bbox_area[2]]
        # depth inside mask of the bbox
        depth_bbox_crop_mask = depth_metric[mask_bool]   
        # depth obs (w/ valid depth)
        depth_bbox_crop_mask_valid = depth_bbox_crop_mask[depth_bbox_crop_mask>0]
        # reshape to vector
        depth_obs_foreground = depth_bbox_crop_mask_valid.reshape(-1)
        # get the rays for those valid depth values in foreground
        # TODO: how to mask the same area as depth
        # get a mask from bbox to mask with valid depth
        mask_and_valid_depth = mask_bool * (depth_metric>0)
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
        mask_inside_bbox = np.full_like(mask_bool, False)
        mask_inside_bbox[bbox_area[1]:bbox_area[3], bbox_area[0]:bbox_area[2]] = True
        mask_inside_bbox_not_mask = (mask_inside_bbox) * (~mask_bool)
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

        return rays_output.astype(np.float32), depth_valid.astype(np.float32)


    def get_t_obj_deepsdf(self, object):
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
        t_obj_deepsdf[0:3, 0:3] = np.eye(3, dtype=np.float32)
        t_obj_deepsdf[0,0] *= object.estimated_scale[0]
        t_obj_deepsdf[1,1] *= object.estimated_scale[1]
        t_obj_deepsdf[2,2] *= object.estimated_scale[2]

        return t_obj_deepsdf