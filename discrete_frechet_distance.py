import similaritymeasures as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

class DiscreteFrechetDistance:

    @staticmethod
    def distance(a, b):
        """
        To calculate the distance matrix between two sets of points.
        :param a: points set A
        :param b: points set B
        :return: the minimal distance between A and B. and the points that generates the min distance
        """
        distances = np.linalg.norm(a - b, axis = 1)
        return np.min(distances), np.argmin(distances)

    @staticmethod
    def distance_of_two_point(a, b):
        """
        Euclidean distance calculation using np linear algebra norm of matrices
        :param a: point a
        :param b: point b
        :return: the distance between a and b
        """
        distance = np.linalg.norm(a - b)
        return distance

    @classmethod
    def distance_matrix(cls, P, Q):
        """
        The distance matrix of point set P and point set Q
        :param P: a point set P (dimension: p * 2)
        :param Q: a point set Q (dimension: q * 2)
        :return: distance matrix.
        """
        pLen = len(P)
        qLen = len(Q)
        distance_matrix = np.zeros((pLen, qLen))
        for i in range(pLen):
            distance_matrix[i, :] = np.linalg.norm(Q - P[i], axis = 1)
        return distance_matrix

    @classmethod
    def discrete_frech_distance(cls, P, Q):
        M = cls.distance_matrix(P, Q)
        pLen, qLen = len(P), len(Q)
        frech_dist = np.zeros((pLen, qLen)) + np.infty
        frech_dist[0, 0] = M[0, 0]
        for j in range(1, qLen):
            frech_dist[0, j] = max(frech_dist[0, j-1], M[0, j])
        for i in range(1, pLen):
            frech_dist[i, 0] = max(frech_dist[i-1, 0], M[i, 0])
            for j in range(1, qLen):
                frech_dist[i, j] = max(min(frech_dist[i-1, j-1],
                                           frech_dist[i-1, j],
                                           frech_dist[i, j-1]),
                                           M[i, j])
        ans_dist = frech_dist[pLen-1, qLen-1]
        ans_pt_indices = np.where(frech_dist == ans_dist)
        ans_pt_index = np.argmin(ans_pt_indices[0] + ans_pt_indices[1])
        # print(M)
        # print(frech_dist)
        # print("ans_dist: ", ans_dist)
        # print("ans_pt_indices[0][ans_pt_index]: ", ans_pt_indices[0][ans_pt_index])
        # print("ans_pt_indices[0]: ", ans_pt_indices[0])
        # print("ans_pt_indices[1][ans_pt_index]: ", ans_pt_indices[1][ans_pt_index])
        # print("ans_pt_indices[1]: ", ans_pt_indices[1])
        return ans_dist, ans_pt_indices[0][ans_pt_index], ans_pt_indices[1][ans_pt_index]

    @classmethod
    def recursive_frech_distance(cls, P, Q):
        pLen, qLen = len(P), len(Q)
        if pLen == qLen == 1:
            distance = cls.distance_of_two_point(P[-1], Q[-1])
            return distance, 0, 0
        elif qLen == 1:
            distances = np.linalg.norm(P - Q[-1], axis=1)
            max_distance, max_index = np.max(distances), np.argmax(distances)
            return max_distance, max_index, 0
        elif pLen == 1:
            distances = np.linalg.norm(Q - P[-1], axis=1)
            max_distance, max_index = np.max(distances), np.argmax(distances)
            return max_distance, 0, max_index
        else:
            res1 = cls.recursive_frech_distance(P[:-1], Q[:-1])
            res2 = cls.recursive_frech_distance(P[:-1], Q)
            res3 = cls.recursive_frech_distance(P, Q[:-1])
            res_array = [res1, res2, res3]
            min_index = int(np.argmin([i[0] for i in res_array]))
            dist_last = cls.distance_of_two_point(P[-1], Q[-1])
            if res_array[min_index][0] > dist_last:
                return res_array[min_index]
            return dist_last, pLen-1, qLen-1



if __name__ == '__main__':
    P = np.array([[1, 1], [2, 2], [2, 2.5], [13, 3], [4, 4], [5, 5]])
    Q = np.array([[-1.1, 1], [1.1, 1], [2.1, 2], [3.1, 3], [4.1, 4], [5.1, 5]])
    Q = Q[::-1]

    f_d, p_i, q_i = DiscreteFrechetDistance.recursive_frech_distance(P, Q)
    print("recursive frechet distance: ", f_d, p_i, q_i)
    f_d, p_i, q_i = DiscreteFrechetDistance.discrete_frech_distance(P, Q)
    print("non frechet distance:  ", f_d, p_i, q_i)
    d = sm.frechet_dist(P, Q)
    print("library frechet distance: ", d)
    dist = directed_hausdorff(P, Q)
    print("hausdorff distance P-Q: ", dist)
    dist = directed_hausdorff(Q, P)
    print("hausdorff distance Q-P: ", dist)
    px = [p[0] for p in P]
    py = [p[1] for p in P]
    qx = [p[0] for p in Q]
    qy = [p[1] for p in Q]
    plt.scatter(px+qx, py+qy)
    plt.plot(px, py, label="P", color="red")
    plt.plot(qx, qy, label="Q", color="blue")
    plt.plot([px[p_i], qx[q_i]], [py[p_i], qy[q_i]], label="frechet dist", linestyle=":", color="purple")
    plt.show()

    df = sm.area_between_two_curves(P, Q)
    print("area bt two curves: {}".format(df))
    df = sm.dtw(P, Q)
    print("dtw: {}".format(df))
    df = sm.curve_length_measure(P, Q)
    print("curve length: {}".format(df))
    df = sm.pcm(P, Q)
    print("pcm: {}".format(df))