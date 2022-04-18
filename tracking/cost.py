from pdb import line_prefix
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
import math

# def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
#     """Computes the generalized intersection over union of two sets of bounding boxes.
#     Followed the math equation in report. 

#     Args:
#         bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
#         bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
#     Returns:
#         g_iou: matrix of shape [M, N], where g_iou[i, j] is the Generalized IoU value between bboxes[i] and bboxes[j].
#     """
#     M, N = bboxes1.shape[0], bboxes2.shape[0]
#     iou_mat = np.zeros((M, N))
#     d_iou = np.zeros((M, N))
#     for i in range(M):
#         bbox_i = bboxes1[i]
#         x_i = bbox_i[0]
#         y_i = bbox_i[1]
#         l_i = bbox_i[2] / 2
#         w_i = bbox_i[3] / 2
#         yaw_i = bbox_i[4] * 180 / math.pi # convert to degree
#         poly_i = Polygon([(x_i-l_i, y_i-w_i),(x_i-l_i, y_i+w_i),(x_i+l_i, y_i+w_i),(x_i+l_i, y_i-w_i)]) # (bottom left, top left, top right, bottom right)
#         rotated_poly_i = affinity.rotate(poly_i, yaw_i, origin='centroid')
#         x_min_i, y_min_i, x_max_i, y_max_i = rotated_poly_i.bounds
#         for j in range(N):
#             bbox_j = bboxes2[j]
#             x_j = bbox_j[0]
#             y_j = bbox_j[1]
#             l_j = bbox_j[2] / 2
#             w_j = bbox_j[3] / 2
#             yaw_j = bbox_j[4] * 180 / math.pi # convert to degree
#             poly_j = Polygon([(x_j-l_j, y_j-w_j),(x_j-l_j, y_j+w_j),(x_j+l_j, y_j+w_j),(x_j+l_j, y_j-w_j)]) # (bottom left, top left, top right, bottom right)
#             rotated_poly_j = affinity.rotate(poly_j, yaw_j, origin='centroid')
#             x_min_j, y_min_j, x_max_j, y_max_j = rotated_poly_j.bounds
#             x_1_c = np.minimum(x_min_i, x_min_j)
#             x_2_c = np.maximum(x_max_i, x_max_j)
#             y_1_c = np.minimum(y_min_i, y_min_j)
#             y_2_c = np.maximum(y_max_i, y_max_j)
#             d_2 = (x_j - x_i)**2 + (y_j - y_i)**2
#             c_2 = (x_2_c - x_1_c)**2 + (y_2_c - y_1_c)**2
#             # print(x_j, x_i, y_j, y_i, x_1_c, x_2_c, y_1_c, y_2_c, d_2, c_2)
#             if rotated_poly_i.intersection(rotated_poly_j).area == 0.0 or rotated_poly_i.union(rotated_poly_j).area == 0.0:
#                 iou_mat[i][j] = 0.0
#             else:
#                 iou_mat[i][j] = rotated_poly_i.intersection(rotated_poly_j).area / rotated_poly_i.union(rotated_poly_j).area 
#                 if c_2 == 0.0:
#                     d_iou[i][j] = iou_mat[i][j]
#                 else:
#                     d_iou[i][j] = iou_mat[i][j] + d_2 / c_2
#     return d_iou


def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    iou_mat = np.zeros((M, N))
    for i in range(M):
        bbox_i = bboxes1[i]
        x_i = bbox_i[0]
        y_i = bbox_i[1]
        l_i = bbox_i[2] / 2
        w_i = bbox_i[3] / 2
        yaw_i = bbox_i[4] * 180 / math.pi # convert to degree
        poly_i = Polygon([(x_i-l_i, y_i-w_i),(x_i-l_i, y_i+w_i),(x_i+l_i, y_i+w_i),(x_i+l_i, y_i-w_i)]) # (bottom left, top left, top right, bottom right)
        rotated_poly_i = affinity.rotate(poly_i, yaw_i, origin='centroid')
        for j in range(N):
            bbox_j = bboxes2[j]
            x_j = bbox_j[0]
            y_j = bbox_j[1]
            l_j = bbox_j[2] / 2
            w_j = bbox_j[3] / 2
            yaw_j = bbox_j[4] * 180 / math.pi # convert to degree
            poly_j = Polygon([(x_j-l_j, y_j-w_j),(x_j-l_j, y_j+w_j),(x_j+l_j, y_j+w_j),(x_j+l_j, y_j-w_j)]) # (bottom left, top left, top right, bottom right)
            rotated_poly_j = affinity.rotate(poly_j, yaw_j, origin='centroid')
            if rotated_poly_i.intersection(rotated_poly_j).area == 0.0 or rotated_poly_i.union(rotated_poly_j).area == 0.0:
                iou_mat[i][j] = 0.0
            else:
                iou_mat[i][j] = rotated_poly_i.intersection(rotated_poly_j).area / rotated_poly_i.union(rotated_poly_j).area 
            # iou_mat[i][j] = poly_i.intersection(poly_j).area / poly_i.union(poly_j).area
    return iou_mat
