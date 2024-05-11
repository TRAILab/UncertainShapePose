
#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Modify from deepsdf / ./DeepSDF/deep_sdf/metrics/chamfer.py

import numpy as np

def get_loss(pts1, pts2, loss_type='chamfer_distance'):
    if loss_type == 'chamfer_distance':
        return compute_trimesh_chamfer(pts1, pts2)
    else:
        raise(NotImplementedError)


import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

def compute_trimesh_chamfer(points1, points2):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    # one direction
    gen_points_kd_tree = KDTree(points1)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(points2)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(points2)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(points1)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer