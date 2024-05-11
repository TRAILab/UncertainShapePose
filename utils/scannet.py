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


'''

Load ScanNet data.

'''

# add sys dir
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import Dataset

from utils.scan2cad import Scan2CAD

import open3d as o3d
import numpy as np

import quaternion
import cv2
# geometry utils

# 1       wall
# 2       floor
# 3       cabinet
# 4       bed
# 5       chair
# 6       sofa
# 7       table
# 8       door
# 9       window
# 10      bookshelf
# 11      picture
# 12      counter
# 14      desk
# 16      curtain
# 24      refridgerator
# 28      shower curtain
# 33      toilet
# 34      sink
# 36      bathtub
# 39      otherfurniture

def change_nyu40ids_to_name(id):
    nyu40ids_to_name = {
        1: 'wall',
        2: 'floor',
        3: 'cabinet',
        4: 'bed',
        5: 'chair',
        6: 'sofa',
        7: 'table',
        8: 'door',
        9: 'window',
        10: 'bookshelf',
        11: 'picture',
        12: 'counter',
        14: 'desk',
        16:      'curtain',
        24:      'refridgerator',
        28:      'shower curtain',
        33:      'toilet',
        34:      'sink',
        36:      'bathtub',
        39:      'otherfurniture'

    }

    if id in nyu40ids_to_name:
        return nyu40ids_to_name[id]
    else:
        return None

# define shape net cat to name
shapenet_category_to_name = {
    '04379243': 'table',
    '03593526': 'jar',
    '04225987': 'skateboard',
    '02958343': 'car',
    '02876657': 'bottle',
    '04460130': 'tower',
    '03001627': 'chair',
    '02871439': 'bookshelf',
    '02942699': 'camera',
    '02691156': 'airplane',
    '03642806': 'laptop',
    '02801938': 'basket',
    '04256520': 'sofa',
    '03624134': 'knife',
    '02946921': 'can',
    '04090263': 'rifle',
    '04468005': 'train',
    '03938244': 'pillow',
    '03636649': 'lamp',
    '02747177': 'trash bin',
    '03710193': 'mailbox',
    '04530566': 'watercraft',
    '03790512': 'motorbike',
    '03207941': 'dishwasher',
    '02828884': 'bench',
    '03948459': 'pistol',
    '04099429': 'rocket',
    '03691459': 'loudspeaker',
    '03337140': 'file cabinet',
    '02773838': 'bag',
    '02933112': 'cabinet',
    '02818832': 'bed',
    '02843684': 'birdhouse',
    '03211117': 'display',
    '03928116': 'piano',
    '03261776': 'earphone',
    '04401088': 'telephone',
    '04330267': 'stove',
    '03759954': 'microphone',
    '02924116': 'bus',
    '03797390': 'mug',
    '04074963': 'remote',
    '02808440': 'bathtub',
    '02880940': 'bowl',
    '03085013': 'keyboard',
    '03467517': 'guitar',
    '04554684': 'washer',
    '02834778': 'bicycle',
    '03325088': 'faucet',
    '04004475': 'printer',
    '02954340': 'cap'
}

# Malisiewicz et al.
def nms_fast(boxes, scores, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(scores)#[::-1]
	#idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return pick


def transform_mat(trs_data, func='official'):
    '''
    @return: 4x4 transformation matrix
    '''
    trans = trs_data['translation']
    quat = trs_data['rotation'] # [qw, qx, qy, qz], // <-- rotation quaternion
    scale = trs_data['scale']

    if func == 'official':
        return make_M_from_tqs(trans, quat, scale)
    else:
        # wxyz to xyzw
        quat = [quat[1], quat[2], quat[3], quat[0]]

        t_world_obj = np.eye(4)
        t_world_obj[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
        t_world_obj = np.matmul(t_world_obj, np.diag([scale[0], scale[1], scale[2], 1]))
        t_world_obj[:3, 3] = trans

        return t_world_obj

def make_M_from_tqs(t, q, s):
    '''
    Function from official Scan2CAD repo
    '''
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 

# geometry utils
def project_point_to_image_batch(point_w_batch, t_world_cam, K):
    '''
    Project a point to image plane
    @param point: 3D point in world coordinate, (N, 3)
    @param t_world_cam: 4x4 transformation matrix from world to camera
    @param K: camera intrinsic matrix

    @return point_img: 2D point in image plane, (2,)
    '''
    # transform to camera coordinate
    t_cam_world = np.linalg.inv(t_world_cam)
    point_cam = np.matmul(t_cam_world, np.concatenate([point_w_batch, np.ones((point_w_batch.shape[0], 1))], axis=1).transpose())
    point_cam = point_cam[:3, :] / point_cam[3, :]

    # project to image plane
    point_img = np.matmul(K, point_cam)

    # make sure the point is in front of the camera
    mask_point_img = point_img[2, :] > 0
    # if not infront, set to negative
    point_img[:, ~mask_point_img] = -1
    # else, normalize to positive
    point_img[:, mask_point_img] = point_img[:, mask_point_img] / point_img[2, mask_point_img]
    # point_img = point_img / point_img[2]

    return point_img.transpose()[:, :2]

# geometry utils
def project_point_to_image(point_w, t_world_cam, K):
    '''
    Project a point to image plane
    @param point: 3D point in world coordinate
    @param t_world_cam: 4x4 transformation matrix from world to camera
    @param K: camera intrinsic matrix

    @return point_img: 2D point in image plane, (2,)
    '''
    # transform to camera coordinate
    t_cam_world = np.linalg.inv(t_world_cam)
    point_cam = np.matmul(t_cam_world, np.concatenate([point_w, [1]]))
    point_cam = point_cam[:3] / point_cam[3]

    # project to image plane
    point_img = np.matmul(K, point_cam)

    # make sure the point is in front of the camera
    if point_img[2] < 0:
        return None

    point_img = point_img / point_img[2]

    return point_img[:2]


def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M

class Frame:
    '''
    Load basic :
    Frame(rgb, depth, t_world_cam, K)
    '''
    def __init__(self, frame_id, rgb, depth, mask, t_world_cam, K) -> None:
        self.frame_id = frame_id
        self.rgb = rgb
        self.depth = depth
        self.t_world_cam = t_world_cam
        self.K = K

        # scores, labels, masks, bboxes
        if mask is not None:
            self.labels_raw = mask['labels']
            self.scores_raw = mask['scores']
            self.bboxes_raw = mask['bboxes']
            self.masks_raw = mask['masks']

            pick = nms_fast(self.bboxes_raw, self.scores_raw, 0.7)

            self.labels = self.labels_raw[pick]
            self.scores = self.scores_raw[pick]
            self.bboxes = self.bboxes_raw[pick,:]
            self.masks = self.masks_raw[pick,:,:]

            self.n_bboxes = len(self.labels)

        self.n_obs = 0

        self.observations = []

    def add_observation(self, observation):
        self.observations.append(observation)
        self.n_obs += 1

class ScanNet(Dataset):
    '''
    Initialize ScanNet, with Scan2CAD and ShapeNet
    '''

    def __init__(self, root_dir, scan2cad_dir=None, shapenet_dir=None) -> None:

        self.root_dir = root_dir

        self.load_data()

        # init scan2cad dataset
        if scan2cad_dir is None:
            scan2cad_dir = os.path.join(self.root_dir, 'data', 'Scan2CAD')
        if shapenet_dir is None:
            shapenet_dir = os.path.join(self.root_dir, 'data', 'ShapeNetCore.v2')

        # print('loading Scan2CAD dataset from: ', scan2cad_dir)
        # print('loading ShapeNet dataset from: ', shapenet_dir)
        self.scan2cad = Scan2CAD(scan2cad_dir, shapenet_dir=shapenet_dir)

        # scene_detail_cache
        self.scene_detail_cache = {}

    def load_data(self):
        '''
        Load data from root dir
        '''

        self.dir_posed_images = os.path.join(self.root_dir, 'data', 'posed_images')

        # data/scannet/scannetv2_val.txt
        scene_list_txt = os.path.join(self.root_dir, 'scannetv2_val.txt')
        self.scene_list = open(scene_list_txt).read().splitlines()

        # Load ScanNet origin annotation; to match the instance id with Scan2CAD

    def get_scene_name_list(self):
        return self.scene_list

    def load_objects_from_scene(self, scene_name):
        # scene_posed_images = os.path.join(self.root_dir, 'data', 'posed_images', scene_name)

        # load objects list from Scan2CAD dataset
        object_data_list = self.scan2cad.load_objects_of_scene(scene_name)

        return object_data_list

    def load_objects_orders_from_scene_with_category(self, scene_name, category):
        '''
        @ category: e.g., chair; see shapenet_category_to_name()
        '''

        object_data_list = self.load_objects_from_scene(scene_name)

        # parse the objects, and load its shapenet shape
        object_orders = []
        for order, obj in enumerate(object_data_list):
            # print(obj)
            shape_id = obj['id_cad']
            shape_catid = obj['catid_cad']

            if shape_catid in shapenet_category_to_name and shapenet_category_to_name[shape_catid] == category:
            # if shapenet_category_to_name[shape_catid] == category:
                object_orders.append(order)
            

        return object_orders


    def get_instance_num(self, scene_name):
        '''
        Get the number of instances in a scene
        '''
        return len(self.load_objects_from_scene(scene_name))

    def load_annotation_of_scene(self, scene_name):
        '''
        Load annotation of a scene
        '''
        
        return self.scan2cad.load_annotation_of_scene(scene_name)

    def get_t_world_scan(self, scene_name):
        trs_data = self.scan2cad.load_annotation_of_scene(scene_name)['trs'] # <-- transformation from scan space to world space 

        # scan coordinate to world
        t_world_scan = transform_mat(trs_data)

        return t_world_scan
    

    def load_scene_pts(self, scene_name):
        '''
        output: o3d.Pointcloud: scene_points in world coordinate.
        '''
        # clean_2: sampled points, much smaller
        scene_pts = os.path.join(self.root_dir, 'data', 'scans', scene_name, scene_name + '_vh_clean_2.ply')

        # load ply
        pcd = o3d.io.read_point_cloud(scene_pts)

        trs_data = self.scan2cad.load_annotation_of_scene(scene_name)['trs'] # <-- transformation from scan space to world space 

        # scan coordinate to world
        t_world_scan = transform_mat(trs_data)
        pcd_world = pcd.transform(t_world_scan)

        # Temp debug
        # debug = True
        # if debug:
        #     t_world_scan_origin = transform_mat(trs_data, func='origin')
        #     t_world_scan_official = transform_mat(trs_data, func='official')
        #     print('t_world_scan_origin: ', t_world_scan_origin)
        #     print('t_world_scan_official: ', t_world_scan_official)

        #     print('done')
        
        return pcd_world

    def visualize_scene(self, scene_name, vis=None, save_dir='./output'):
        '''
        Visualize a scene with pointcloud, and annotated objects from Scan2CAD dataset
        '''

        scene_pts = self.load_scene_pts(scene_name)

        # load objects list from Scan2CAD dataset
        objects = self.load_objects_from_scene(scene_name)

        # parse the objects, and load its shapenet shape
        shape_mesh_list = []
        for obj in objects:
            # print(obj)
            shape_id = obj['id_cad']
            shape_catid = obj['catid_cad']
            shape_mesh = self.scan2cad.load_shape(shape_catid, shape_id)

            trs_data = obj['trs']  # from CAD space to world space 
            t_world_obj = transform_mat(trs_data)
            shape_mesh_world = shape_mesh.transform(t_world_obj)

            shape_mesh_list.append(shape_mesh_world)

        # begin visualizing
        if vis is None:
            vis = o3d.visualization.Visualizer()
            vis.create_window()

        vis.clear_geometries()
        # add all meshes
        for mesh in shape_mesh_list:
            vis.add_geometry(mesh)
        vis.add_geometry(scene_pts)
        
        # render in headless mode
        vis.poll_events()
        vis.update_renderer()
        
        # capture image
        image = vis.capture_screen_float_buffer(False)
        image = np.asarray(image) * 255

        # save visualized ply into file
        if save_dir is not None:
            save_path = os.path.join(save_dir, 'scannet_vis', scene_name, 'scene_vis_world')
            os.makedirs(save_path, exist_ok=True)

            # save scene 
            scene_pts_path = os.path.join(save_path, scene_name + '_scene.ply')
            o3d.io.write_point_cloud(scene_pts_path, scene_pts)

            # save objects
            for i, mesh in enumerate(shape_mesh_list):
                obj_path = os.path.join(save_path, scene_name + '_obj_' + str(i) + '.ply')
                o3d.io.write_triangle_mesh(obj_path, mesh)
            
            # visualize the frame poses of the scene
            frame_poses = self._load_frame_poses(scene_name)
            for i, t_world_cam in enumerate(frame_poses):
                pose_path = os.path.join(save_path, scene_name + '_pose_' + str(i) + '.ply')
                pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                pose_mesh_world = pose_mesh.transform(t_world_cam)
                o3d.io.write_triangle_mesh(pose_path, pose_mesh_world)

            # save image
            cv2.imwrite(os.path.join(save_path, scene_name + '.png'), image)

            print('saved visualized scene into: ', save_path)
            
        

        return image    

    def load_object_observations_from_scene(self, scene_name, obj_id, load_mesh=False, load_image=True):
        '''
        @load_mesh: Close it to save time.

        return the content:
        {
            'scene_name': scene_name,
            'obj_id': obj_id,
            'object_mesh': object_mesh,
            'trs': trs,
            't_world_obj': t_world_obj,
            'frames': a list of frames that can see this object. Each frame is a dict:
                {
                    'frame_id': frame_id,
                    'cam_pose': cam_pose,
                    'cam_intrinsics': cam_intrinsics,

                    ...

                }       
        }
        '''

        object_list = self.load_objects_from_scene(scene_name)

        # check id valid
        if obj_id >= len(object_list):
            print('obj_id: ', obj_id, ' is larger than all objects number: ', len(object_list))
            return None

        obj = object_list[obj_id]

        # output dict
        output = {}
        output['scene_name'] = scene_name
        output['obj_id'] = obj_id

        # get object mesh
        if load_mesh:
            object_mesh = self.scan2cad.load_shape(obj['catid_cad'], obj['id_cad'])
            output['object_mesh'] = object_mesh
        else:
            output['object_mesh'] = 'NOT LOADED'

        output['catid_cad'] = obj['catid_cad']
        output['id_cad'] = obj['id_cad']
        
        output['category'] = shapenet_category_to_name[obj['catid_cad']]

        # get object pose
        trs_data = obj['trs']  # from CAD space to world space 
        t_world_obj = transform_mat(trs_data)
        # shape_mesh_world = shape_mesh.transform(t_world_obj)
        output['t_world_obj'] = t_world_obj

        # also save the origin gt information
        trans = trs_data['translation']
        quat = trs_data['rotation'] # [qw, qx, qy, qz], // <-- rotation quaternion
        scale = trs_data['scale']
        output['t_world_obj_trans'] = trans
        output['t_world_obj_quat'] = quat
        output['t_world_obj_scale'] = scale

        # add bbox_t_world_unit
        # output['bbox_t_world_unit'] = calc_Mbbox(obj)

        # add sym
        output['sym'] = obj['sym']

        # collect all frames that can see this object
        # check if the center projected to the image is inside the object mesh?
        # or check if there are annotated points inside the object mesh?
        
        obj_center_world = t_world_obj[:3, 3]
        frames = self._collect_frames_observing_object(scene_name, obj_center_world, load_image=load_image)  
        output['frames'] = frames

        debug = False
        if debug:
            # save the point into ply
            obj_center_pcd = o3d.geometry.PointCloud()
            obj_center_pcd.points = o3d.utility.Vector3dVector(np.expand_dims(obj_center_world, axis=0))
            obj_center_pcd.paint_uniform_color([1, 0, 0])
            o3d.io.write_point_cloud('output/obj_center.ply', obj_center_pcd)

        # load index transform to origin scannet
        index_transform = self.load_index_transform(scene_name)
        output['index_transform'] = index_transform

        '''
        further load annotation points here
        '''
        output['keypoints_cad'] = obj['keypoints_cad']
        output['keypoints_scan'] = obj['keypoints_scan']

        return output

    def load_index_transform(self, scene_name):
        '''
        load index transform from the scene
        '''
        index_transform_path = os.path.join(self.dir_posed_images, scene_name, 'indices.npy')

        # frame_indices.npy
        index_transform_path2 = os.path.join(self.dir_posed_images, scene_name, 'frame_indices.npy')
        
        # check if exist
        if os.path.exists(index_transform_path):
            index_transform = np.load(index_transform_path)    
        elif os.path.exists(index_transform_path2):
            index_transform = np.load(index_transform_path2)  
        else:
            print('index transform not exist: ', index_transform_path, 'and ', index_transform_path2)
            return None
        return index_transform

    def _collect_frames_observing_object(self, scene_name, obj_center_world, load_image=True):
        # collect frame_pose-ids
        frame_poses = self._load_frame_poses(scene_name)

        # get intrinsics
        scene_intrinsics = self.load_scene_intrinsics(scene_name)

        # get image width, height
        image_width, image_height = self.load_image_width_height(scene_name)

        # check projection and whether inside image
        valid_frame_ids = []
        projected_centers = []
        for i, frame_pose in enumerate(frame_poses):
            if frame_pose is None:
                continue # no valid pose

            # proj center to image plane
            center_proj = project_point_to_image(obj_center_world, frame_pose, scene_intrinsics)

            if center_proj is None:
                # it is behind the camera
                continue

            # check if inside image
            if center_proj[0] < 0 or center_proj[0] >= image_width or center_proj[1] < 0 or center_proj[1] >= image_height:
                continue

            valid_frame_ids.append(i)
            projected_centers.append(center_proj)

        # load valid frames
        valid_frames = []
        for t, frame_id in enumerate(valid_frame_ids):
            # load
            frame = self.load_frame(scene_name, frame_id, load_image)

            valid_frames.append(frame)


            debug = False
            if debug:
                # visualize the RGB image with projected center
                rgb_image_vis = frame.rgb.copy()

                center = projected_centers[t]

                rgb_image_vis = cv2.circle(rgb_image_vis, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)

                cv2.imwrite(f'output/debug_proj_rgb_f{frame_id}.png', rgb_image_vis)



        return valid_frames

    def load_frame(self, scene_name, frame_id, load_image=True):
        if load_image:
            # load RGB image, Depth image, Cam Pose, Cam Intrinsics
            # load RGB image
            rgb_name = os.path.join(self.dir_posed_images, scene_name, str(frame_id).zfill(5) + '.jpg')
            rgb = cv2.imread(rgb_name)

            # load depth image
            depth_name = os.path.join(self.dir_posed_images, scene_name, str(frame_id).zfill(5) + '.png')
            depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)

            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation= cv2.INTER_NEAREST)

            # Never load mask
            # mask = self.load_detection_mask(scene_name, frame_id, 'mask2former')
            mask = None

        else:
            rgb = None
            depth = None
            mask = None

        # load cam pose
        t_world_cam = self.load_frame_pose(scene_name, frame_id)

        # load cam intrinsics
        K = self.load_scene_intrinsics(scene_name)

        frame = Frame(frame_id, rgb, depth, mask, t_world_cam, K)

        return frame
    
    def load_detection_mask(self, scene_name, frame_id, mask_source='mask2former'):
        '''
        detection mask dir: self.dir_detection_masks
        '''
        self.dir_detection_masks = os.path.join(self.root_dir, 'data', 'detection_masks')

        dir_detection_masks_method = os.path.join(self.dir_detection_masks, mask_source)
        if not os.path.exists(dir_detection_masks_method):
            raise ValueError('detection mask method not exist: ', mask_source)
        
        # load file .pth
        mask_path = os.path.join(dir_detection_masks_method, scene_name, str(frame_id).zfill(5) + '.pth')
        mask = torch.load(mask_path) # scores, labels, masks, bboxes
        #print(mask)

        return mask

    def _load_frame_poses(self, scene_name):
        num_frames = self.load_images_num(scene_name)

        cam_poses_list = []
        for i in range(num_frames):
            # 00000.txt, 00001.txt, ...
            # 0.586492 0.380183 -0.715184 6.016166
            # 0.809684 -0.298032 0.505558 1.433055
            # -0.020943 -0.875579 -0.482622 1.319307
            # 0.000000 0.000000 0.000000 1.000000

            t_world_cam = self.load_frame_pose(scene_name, i)

            cam_poses_list.append(t_world_cam)

        return cam_poses_list

    def load_frame_pose(self, scene_name, i):
        # get t_world_scan
        trs_data = self.scan2cad.load_annotation_of_scene(scene_name)['trs'] # <-- transformation from scan space to world space 
        # scan coordinate to world
        t_world_scan = transform_mat(trs_data)

        frame_pose_file = os.path.join(self.dir_posed_images, scene_name, str(i).zfill(5) + '.txt')

        # if not exist
        if not os.path.exists(frame_pose_file):
            print('frame_pose_file not exist: ', frame_pose_file)
            return None

        t_scan_cam = np.loadtxt(frame_pose_file)  # t_scan_cam

        t_world_cam = t_world_scan @ t_scan_cam  # t_world_cam

        # check if there is nan; # Note those invalid frames has been filtered by the package
        # if np.isnan(t_world_cam).any():
            # print('nan in t_world_cam.')

        # t_world_cam = np.linalg.inv(t_world_cam) # invesre

        return t_world_cam

    def load_scene_intrinsics(self, scene_name):
        '''
        Return a calibration matrix:

        fx 0 cx
        0 fy cy
        0 0 1

        '''

        # option 1: load from render output
        'data/scannet/data/posed_images/scene0645_02/intrinsic.txt'
        # filename = os.path.join(self.dir_posed_images, scene_name, 'intrinsic.txt')

        '''
        Example:

        1169.621094 0.000000 646.295044 0.000000
        0.000000 1167.105103 489.927032 0.000000
        0.000000 0.000000 1.000000 0.000000
        0.000000 0.000000 0.000000 1.000000

        Note the values are a little difference from option 2.
        '''
        # K = np.loadtxt(filename)
        # K = K[:3, :3]


        # option 2: load from origin file
        scene_details = self.load_scene_details(scene_name)

        K = np.array([[scene_details['fx_color'], 0, scene_details['mx_color']],
                        [0, scene_details['fy_color'], scene_details['my_color']],
                        [0, 0, 1]])

        return K
    
    def load_image_width_height(self, scene_name):
        scene_details = self.load_scene_details(scene_name)

        return scene_details['colorWidth'], scene_details['colorHeight']

    def load_images_num(self, scene_name=None, option='current'):
        '''
        current rendering: only 300 frames

        all: all frames

        '''

        if option == 'current':
            return 300
        else:
            scene_details = self.load_scene_details(scene_name)

            # check color and depth are equal
            assert scene_details['numColorFrames'] == scene_details['numDepthFrames']

            return scene_details['numColorFrames']


    def load_scene_details(self, scene_name):
        # data/scannet/data/scans/scene0011_00/scene0011_00.txt

        # axisAlignment = -0.782608 0.622515 0.000000 0.118658 -0.622515 -0.782608 0.000000 5.536450 0.000000 0.000000 1.000000 -0.079202 0.000000 0.000000 0.000000 1.000000 
        # colorHeight = 480
        # colorWidth = 640
        # depthHeight = 480
        # depthWidth = 640
        # fx_color = 577.870605
        # fx_depth = 577.870605
        # fy_color = 577.870605
        # fy_depth = 577.870605
        # mx_color = 319.500000
        # mx_depth = 319.500000
        # my_color = 239.500000
        # my_depth = 239.500000
        # numColorFrames = 1598
        # numDepthFrames = 1598
        # numIMUmeasurements = 3354
        # sceneType = Conference Room

        if scene_name in self.scene_detail_cache:
            return self.scene_detail_cache[scene_name]

        scene_detail_file = os.path.join(self.root_dir, 'data', 'scans', scene_name, scene_name + '.txt')

        scene_detail = {}
        with open(scene_detail_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                tokens = line.split(' = ')
                # scene_detail[tokens[0]] = tokens[1]

                # if they are ints, change to ints; if floats, change to floats; if array, change numpy.array
                if tokens[0] in ['colorHeight', 'colorWidth', 'depthHeight', 'depthWidth', 'numColorFrames', 'numDepthFrames', 'numIMUmeasurements']:
                    scene_detail[tokens[0]] = int(tokens[1])
                elif tokens[0] in ['axisAlignment']:
                    scene_detail[tokens[0]] = np.array([float(x) for x in tokens[1].split(' ')])
                elif tokens[0] in ['fx_color', 'fx_depth', 'fy_color', 'fy_depth', 'mx_color', 'mx_depth', 'my_color', 'my_depth']:
                    scene_detail[tokens[0]] = float(tokens[1])
                else:
                    scene_detail[tokens[0]] = tokens[1]

        
        self.scene_detail_cache[scene_name] = scene_detail

        return scene_detail



    def __len__(self) -> int:
        return self.load_images_num()

    def __getitem__(self, index: int):
        pass


    # dataset helper functions for visualization
    def visualize_object_observations(self, obj_obs, save_dir='./output/scannet_vis'):
        scene_name = obj_obs['scene_name']

        # visualize 3d scene, and the object mesh
        scene_pts = self.load_scene_pts(scene_name)  # defaultly, change to world coordinate now

        # transform mesh to world
        obj_mesh = obj_obs['object_mesh']
        obj_mesh_world = obj_mesh.transform(obj_obs['t_world_obj'])

        # visualize
        # if vis is None:
        #     vis = o3d.visualization.Visualizer()
        #     vis.create_window()
        # vis.add_geometry(scene_pts)
        # vis.add_geometry(obj_mesh_world)
        
        # save to ply
        obj_id = obj_obs['obj_id']
        save_dir_vis = os.path.join(save_dir, scene_name, f'object-{obj_id}')
        os.makedirs(save_dir_vis, exist_ok=True)
        
        # save ply
        # o3d.io.write_point_cloud(os.path.join(save_dir_vis, 'scene.ply'), scene_pts)
        o3d.io.write_triangle_mesh(os.path.join(save_dir_vis, 'object.ply'), obj_mesh_world)

        # visualize camera poses as ply
        frames = obj_obs['frames']
        print('len frames:', len(frames))

        if len(frames) == 0:
            print('no frames found for this object')
            return

        for f in frames:
            cam_pose = f.t_world_cam

            coor_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            coor_mesh_world = coor_mesh.transform(cam_pose)
            
            # save
            o3d.io.write_triangle_mesh(os.path.join(save_dir_vis, f'frame-{f.frame_id}.ply'), coor_mesh_world)

        # visualize a coordinate frame with open3d
        
        # load all RGB images and depth images
        rgb_images = []
        depth_images = []
        for f in frames:
            rgb = f.rgb.copy()
            depth = f.depth.copy()

            # add frame id to image
            frame_id = f.frame_id
            rgb = cv2.putText(rgb, str(frame_id), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            depth = cv2.putText(depth, str(frame_id), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # resize image into 1/4
            rgb = cv2.resize(rgb, (rgb.shape[1] // 4, rgb.shape[0] // 4))
            depth = cv2.resize(depth, (depth.shape[1] // 4, depth.shape[0] // 4))

            rgb_images.append(rgb)
            depth_images.append(depth)

        # visualize in a large image, with index indicating the frame id
        # rgb_images_im = np.stack(rgb_images, axis=0)
        # depth_images_im = np.stack(depth_images, axis=0)

        # compose multiple images into one
        rgb_images_im = cv2.vconcat(rgb_images)
        depth_images_im = cv2.vconcat(depth_images)

        # make depth as RGB
        # normalize depth to 255
        depth_images_im = (depth_images_im / depth_images_im.max() * 255).astype(np.uint8)
        depth_images_im = cv2.cvtColor(depth_images_im, cv2.COLOR_GRAY2RGB)
        # make the height of depth image the same as rgb image
        depth_images_im = cv2.resize(depth_images_im, (rgb_images_im.shape[1], rgb_images_im.shape[0]))

        # concat rgb and depth into one
        rgb_depth_images_im = cv2.hconcat([rgb_images_im, depth_images_im])
        
        # save to dir
        cv2.imwrite(os.path.join(save_dir_vis, 'rgb_depth.png'), rgb_depth_images_im)      
        
        print('save to dir:', save_dir_vis)

# add a unit test function
def test_scene_vis():
    '''
    Render a 3d image with open3d, with scene pts and objects mesh.
    Also save ply files into dir: ./output/scannet_vis
    '''    
    dataset = ScanNet(root_dir='data/scannet')

    # visualize a scene with pointcloud, and annotated objects from Scan2CAD dataset
    image = dataset.visualize_scene('scene0568_00')

    import cv2
    cv2.imwrite('test_vis_scannet.png', image)

# test getting observation of an instance in a scene
def test_object_observations():
    dataset = ScanNet(root_dir='data/scannet')

    # scene_objs = dataset.load_objects_from_scene('scene0568_00')

    # object instance in the scene
    scene_name = 'scene0568_00'
    object_list = dataset.load_objects_from_scene(scene_name)

    # print the object-ins and category of the scene
    print('scene_name:', scene_name)
    for i, obj_data in enumerate(object_list):
        obj_cat = obj_data['catid_cad']
        obj_cat_name = shapenet_category_to_name[obj_cat]
        print(f'object {i}: {obj_cat_name}')

    # given an object id, find all associated observations 
    # (Frames: RGB images, depth images, and camera poses; GTs: object pose, shape)
    for object_id in [5,7,8,9,10]: # all chairs

        object_obs = dataset.load_object_observations_from_scene(scene_name, object_id)

        # 1) 3D: visualize a scene with pointcloud, and annotated objects from Scan2CAD dataset, and camera poses
        # 2) 2D: visualize all RGB images, depth images
        dataset.visualize_object_observations(object_obs, save_dir='./output/scannet_vis')

        print('finish visualizing object observations', object_id)

def test_projection():
    # project the scene point cloud into the camera

    scene_name = 'scene0568_00'

    dataset = ScanNet(root_dir='data/scannet')
    
    # frame_list = range(len(dataset))
    frame_list = [87]

    # load scene point cloud
    scene_pts_o3d_world = dataset.load_scene_pts(scene_name)
    # o3d type to numpy with x,y,z,r,g,b
    scene_pts_world = np.asarray(scene_pts_o3d_world.points)
    scene_pts_color = np.asarray(scene_pts_o3d_world.colors) * 255
    # RGB to BGR
    scene_pts_color = scene_pts_color[:, [2, 1, 0]]

    for frame_id in frame_list:
        print('processing frame_id:', frame_id)

        # load camera pose
        frame = dataset.load_frame(scene_name, frame_id)
        t_world_cam = frame.t_world_cam

        # project scene pts into camera
        scene_pts_cam = np.linalg.inv(t_world_cam) @ np.concatenate([scene_pts_world, np.ones((scene_pts_world.shape[0], 1))], axis=1).T
        scene_pts_cam = scene_pts_cam.T

        scene_pts_cam = scene_pts_cam[:, :3]

        # project into image plane
        K = frame.K
        scene_pts_img = K @ scene_pts_cam.T
        scene_pts_img = scene_pts_img.T
        # only keep valid points
        valid_indices = scene_pts_img[:, 2] > 0
        scene_pts_img_valid = scene_pts_img[valid_indices, :]
        scene_pts_img_valid_uv = scene_pts_img_valid[:, :2] / scene_pts_img_valid[:, 2:]

        # save to png
        scene_pts_color_valid = scene_pts_color[valid_indices, :]
        proj_im = np.zeros_like(frame.rgb)

        valid_count = 0
        for i,pt in enumerate(scene_pts_img_valid_uv):
            pt = pt.astype(np.int32)

            if pt[0] >= 0 and pt[0] < proj_im.shape[1] and pt[1] >= 0 and pt[1] < proj_im.shape[0]:
                # proj_im[pt[1], pt[0], :] = scene_pts_color_valid[i, :]

                # the point cloud is too sparse and can not see clearly
                # we also make the point size larger
                k = 3
                if pt[0] >= k and pt[0] < proj_im.shape[1]-k and pt[1] >= k and pt[1] < proj_im.shape[0]-k:
                    proj_im[pt[1]-k:pt[1]+k, pt[0]-k:pt[0]+k, :] = scene_pts_color_valid[i, :]                
                
                valid_count = valid_count + 1

        print('valid_count:', valid_count)

        cv2.imwrite(f'output/scannet_vis/projection/proj_scene_{scene_name}_f{frame_id}.png', proj_im)

        # over lay the projected image with the rgb image
        rgb_vis = frame.rgb.copy()
        rgb_vis_overlay = cv2.addWeighted(rgb_vis, 0.5, proj_im, 0.5, 0)
        cv2.imwrite(f'output/scannet_vis/projection/proj_scene_{scene_name}_f{frame_id}_overlay.png', rgb_vis_overlay)



def test_scene_object_instance_shapenet():
    '''
    Load all instances of one specific category of all the scenes.
    ''' 

    # data/scannet/data/Scan2CAD/unique_cads.csv
    # import pandas as pd
    # csv_data = pd.read_csv('data/scannet/data/Scan2CAD/unique_cads.csv')

    # load csv as string
    csv_data = np.genfromtxt('data/scannet/data/Scan2CAD/unique_cads.csv', delimiter=',', dtype=str)

    # catid-cad,id-cad
    # 04256520,e877f5881c54551a2665f68ebc337f05
    # 04256520,267dcf5952e84873fad6a32f56e259a2
    # 04256520,918ae8afec3dc70af7628281ecb18112
    # 04379243,3c899eb5c23784f9febad4f49b26ec52
    # 02747177,5a947a76bf8f29035b17dbcbc75d58df
    # 04379243,3eea280aa5da733e95166372d7ac113b
    # 04004475,99df53ab6417e293b3b8d66c43b5b940
    
    # load all 03001627 data

    # get an array of catid-cad, and another of id-cad
    # catid_cad = csv_data['catid-cad'].values
    # id_cad = csv_data['id-cad'].values
    catid_cad = csv_data[1:, 0]
    id_cad = csv_data[1:, 1]

    ins_list_chairs = id_cad[catid_cad == '03001627']

    print('len of ins_list_chairs:', len(ins_list_chairs))

    # TODO preprocess them to get the transformation 

    print('done')

if __name__ == '__main__':
    # test_scene_vis()

    # test_object_observations()

    # test_projection()

    test_scene_object_instance_shapenet()
