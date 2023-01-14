"""
To simplify and speed up the clustering process, the Ramer-Douglas-Peucker algorithm is implemented here.
The algorithm reduces a trajectory to a polyline with fewer points, with precision.
The algorithm is adopted as a preprocessing step.
"""

import math
import numpy as np

def euclidean_distance(m, n):
    return math.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2)

def point_distance_to_line(current_pt, start_pt, destination_pt):
    if start_pt == destination_pt:
        return euclidean_distance(current_pt, start_pt)
    else:
        m = abs((destination_pt[0] - start_pt[0]) * (start_pt[1] - current_pt[1])
                     - (start_pt[0] - current_pt[0]) * (destination_pt[1] - start_pt[1]))
        l = math.sqrt((destination_pt[0] - start_pt[0]) ** 2 + (destination_pt[1] - start_pt[1]) ** 2)
        return m / l;

def rdp_reduction(pts, epsilon):
    """

    :param pts: the pts to be reduced using RDP algorithm
    :param epsilon: the limit which represents to which degree we want the original trajectory to be reduced
    :return: the polyline denoted by a list of points, resulting from the rdp reduction.
    """
    max_idx = 0
    max_distance = 0.0

    for i in range(1, len(pts) - 1):
        dst = point_distance_to_line(pts[i], pts[0], pts[-1])
        if dst > max_distance:
            max_idx = i
            max_distance = dst

    if max_distance > epsilon:
        ans = rdp_reduction(pts[:max_idx + 1], epsilon)[:-1] + rdp_reduction(pts[max_idx:],epsilon)
    else:
        ans = [pts[0], pts[-1]]

    return ans

def indexed_rdp_reduction(pts, idces, epsilon):
    max_idx = 0
    max_distance = 0.0

    for i in range(1, len(pts) - 1):
        dst = point_distance_to_line(pts[i], pts[0], pts[-1])
        if dst > max_distance:
            max_idx = i
            max_distance = dst

    if max_distance > epsilon:
        front_pts, front_idces = indexed_rdp_reduction(pts[:max_idx+1], idces[:max_idx+1], epsilon)
        back_pts, back_idces = indexed_rdp_reduction(pts[max_idx:], idces[max_idx:], epsilon)
        pts_ans = front_pts[:-1] + back_pts
        idces_ans = front_idces[:-1] + back_idces
    else:
        pts_ans = [pts[0], pts[-1]]
        idces_ans = [idces[0], idces[-1]]

    return pts_ans, idces_ans

if __name__ == "__main__":
    test_pts = [[0, 0], [100, 100], [230, 200], [220, 210], [350, 350], [450, 450]]
    rdp_reduction_ans_pts = rdp_reduction(test_pts, epsilon=20)
    print(np.array(rdp_reduction_ans_pts))
    indexed_rdp_reduction_pts, indexed_rdp_reduction_indices = indexed_rdp_reduction(test_pts, list(range(len(test_pts))), epsilon=20)
    print(indexed_rdp_reduction_pts)
    print(indexed_rdp_reduction_indices)
