'''

An example loader to load data from ScanNet.

'''
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.scannet import ScanNet
import cv2
import open3d as o3d

import matplotlib.pyplot as plt
import  numpy as np

from data_association.association_utils import *

def visualize_mask(frame):

    # plot bboxes on rgb
    rgb_vis = frame.rgb.copy()
    for obs in frame.observations:

        rgb_vis = cv2.rectangle(rgb_vis, obs.bbox[0:2], obs.bbox[2:4], (0, 255, 0), 2)

        # score only keep 2 digits
        text = '{}: {:.2f}'.format(obs.name_class, obs.score)
        rgb_vis = cv2.putText(rgb_vis, text, obs.bbox[0:2], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # also plot masks, indicating with different color
        mask = obs.mask_inflated.astype(bool)
        rgb_vis[mask] = 0.5 * rgb_vis[mask] + 0.5 * np.array(obs.coco_color)

    # rgb to bgr
    #rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
    im_save_name = f'vis/rgb_bboxes_f{frame.frame_id}.png'
    cv2.imwrite(im_save_name, rgb_vis)
    #print('  save vis to: ', im_save_name)


def main():
    # Select the mask source.
    # "mask2former": use an off-the-shelf object detection results
    # "gt": Offered by the dataset. projecting the GT segmented semantic point cloud into the image plane. 
    mask_source = 'mask2former'  
    
    ###
    # load scannet
    dataset = ScanNet(root_dir='data/scannet')

    # select a scene_name
    scene_name = 'scene0568_00'

    bad_frames = [8, 9, 16, 17, 21, 22, 24, 25, 26, 29, 194, 209, 250, 264]

    # go through all frames
    # scene_info = dataset.load_annotation_of_scene(scene_name)
    # num_frames = scene_info['num_frames']
    num_frames = dataset.load_images_num(scene_name)

    # there are 300 frames per scene. we can consider every 10 frames.
    my_scene = Scene()
    skip = 1
    for frame_id in range(0, num_frames, skip):
    #for frame_id in range(0,20):
        if frame_id in bad_frames:
            continue
        print("Working on frame:",frame_id)
        # load frame information
        frame = dataset.load_frame(scene_name, frame_id)

        for j in range(frame.n_bboxes):
            label = frame.labels[j]
            if not coco_id_in_intereted_classes(label) or frame.scores[j] < 0.6:
                continue
            obs = Observation(frame, j)
            frame.add_observation(obs)
        print("Number of observations:",frame.n_obs)

        # visualize current frame of rgb,depth
        #visualize_mask(frame)

        if (frame.n_obs == 0):
            continue
        
        #visualize_frame_obs(frame)
        my_scene.add_frame(frame)

    my_scene.prune_overlapping_objects()
    my_scene.estimate_poses()
    my_scene.visualize_objects()
    

    # TODO: return data association structure


    # load 3d scene points in world frame for visualization
    #scene_pts_world = dataset.load_scene_pts(scene_name)

    print('DONE')


if __name__ == '__main__':
    main()