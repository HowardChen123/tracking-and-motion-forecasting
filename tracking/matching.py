from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def greedy_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the greedy matching algorithm.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    row_ids = []
    col_ids = []
    M, N = cost_matrix.shape
    S_1 = [m for m in range(M)]
    S_2 = [n for n in range(N)]
    min_val = min(M, N)
    # sorted to find least cost C(i, j) lists
    sorted = np.unique(np.sort(cost_matrix, axis=None))
    # realize min(M,N) iterations
    count = 0
    i = 0
    while count < min_val + 1 and i < sorted.shape[0]:
        index = np.argwhere(cost_matrix == sorted[i])
        len_ind = index.shape[0]
        for j in range(len_ind):
            if index[j][0] in S_1 and index[j][1] in S_2:
                row_ids.append(index[j][0])
                col_ids.append(index[j][1]) 
                S_1.remove(index[j][0])
                S_2.remove(index[j][1])
                count += 1
        i += 1
    return row_ids, col_ids


def hungarian_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the Hungarian matching algorithm.
    For simplicity, we just call the scipy `linear_sum_assignment` function. Please refer to
    https://en.wikipedia.org/wiki/Hungarian_algorithm and
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
    for more details of the hungarian matching implementation.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    # row_ids = [], col_ids = []
    return row_ids, col_ids
