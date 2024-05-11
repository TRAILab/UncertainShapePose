##########################
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

import numpy as np
from hungarian_algorithm import algorithm
from scipy.optimize import linear_sum_assignment


def hungarian_matcher_sp(cost_matrix):
    obs_ind, assoc_obj_ind = linear_sum_assignment(cost_matrix)
    return obs_ind, assoc_obj_ind

def hungarian_matcher(cost_matrix):

    candidate_mask = np.any(cost_matrix, axis=1)
    unmatched_mask = np.any(cost_matrix, axis=0)

    G_dict = {}
    for i in range(cost_matrix.shape[0]):
        if not candidate_mask[i]:
            continue
        G_dict[str(i+1)] = {}
        for j in range(cost_matrix.shape[1]):
            if not unmatched_mask[j]:
                continue
            G_dict[str(i+1)][str(-j-1)] = cost_matrix[i,j]
    print("cost_matrix:",cost_matrix)
    print("G_dict:",G_dict)

    result = []
    if len(G_dict) > 1:
        result = algorithm.find_matching(G_dict, matching_type = 'min', return_type = 'list' )

    if len(G_dict) == 1:
        matched = np.argmax(cost_matrix[0,:])
        matched_score = cost_matrix[0, matched]
        result.append( ((str(i+1), str(-matched-1)), matched_score) )

    for i in range(cost_matrix.shape[0]):
        if not candidate_mask[i]:
            result.append((str(i+1), -1))
    print("Hungarian Result:",result)

    return result

if __name__ == "__main__":
    G_dict = {'1': {'-1': 0, '-2': 0, '-3': 0.9, '-4': 0.8}, '2': {'-1': 0, '-2': 0, '-3': 0.8, '-4': 0.9}}
    print(G_dict)
    result = algorithm.find_matching(G_dict, matching_type = 'max', return_type = 'list' )
    print(result)