from pdb import line_prefix
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
import math

def nms_iou(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Combined the ntersection over union of two sets of bounding boxes and Non-maximum suppression (NMS) to solve the problem of overlapping and occlusion case.
    Followed the math equation in report. 

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou: matrix of shape [M, N], where g_iou[i, j] is the Generalized IoU value between bboxes[i] and bboxes[j].
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    iou_mat = []
    for i in range(M):
        bbox_i = bboxes1[i]
        x_i = bbox_i[0]
        y_i = bbox_i[1]
        l_i = bbox_i[2] / 2
        w_i = bbox_i[3] / 2
        yaw_i = bbox_i[4] * 180 / math.pi # convert to degree
        poly_i = Polygon([(x_i-l_i, y_i-w_i),(x_i-l_i, y_i+w_i),(x_i+l_i, y_i+w_i),(x_i+l_i, y_i-w_i)]) # (bottom left, top left, top right, bottom right)
        rotated_poly_i = affinity.rotate(poly_i, yaw_i, origin='centroid')
        iou_i = []
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
                iou_i.append([0.0, i, j])
            else:
                iou_i.append([rotated_poly_i.intersection(rotated_poly_j).area / rotated_poly_i.union(rotated_poly_j).area, i, j])
        sorted_iou_i = sorted(iou_i, key=lambda x: x[0])
        # iou_mat[i][j] = poly_i.intersection(poly_j).area / poly_i.union(poly_j).area
        iou_mat.append([sorted_iou_i[0][1], sorted_iou_i[0][2]])
    return iou_mat