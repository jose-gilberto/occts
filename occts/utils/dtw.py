""" _summary_
"""
import numpy as np
# import math
# import sys

RETURN_VALUE = 0
RETURN_PATH = 1
RETURN_ALL = -1


def _cummulative_matrix(
    cost: np.ndarray,
    slope_constraint: str,
    window: int
) -> np.ndarray:
    p = cost.shape[0]
    s = cost.shape[1]

    dtw_ = np.full((p + 1, s + 1), np.inf)

    dtw_[0, 0] = 0.0

    if slope_constraint == 'asymmetric':
        for i in range(1, p + 1):
            if i < window + 1:
                dtw_[i, 1] = (
                    cost[i - 1, 0] +
                    min(dtw_[i - 1, 0], dtw_[i - 1, 1])
                )

            for j in range(max(2, i - window), min(s, i + window) + 1):
                dtw_[i, j] = (
                    cost[i - 1, j - 1] +
                    min(dtw_[i - 1, j - 2], dtw_[i - 1, j - 1], dtw_[i - 1, j])
                )

    elif slope_constraint == 'symmetric':
        for i in range(1, p + 1):
            for j in range(max(1, i - window), min(s, i + window) + 1):
                dtw_[i, j] = (
                    cost[i - 1, j - 1] +
                    min(dtw_[i - 1, j - 1], dtw_[i, j - 1], dtw_[i - 1, j])
                )

    else:
        raise ValueError(f'Unknown slope constraint {slope_constraint}')

    return dtw_


def _traceback(dtw_matrix: np.ndarray, slope_constraint: str) -> np.ndarray:
    i, j = np.array(dtw_matrix.shape) - 1
    p, q = [i - 1], [j - 1]

    if slope_constraint == 'asymmetric':
        while i > 1:
            tb = np.argmin((
                dtw_matrix[i - 1, j],
                dtw_matrix[i - 1, j - 1],
                dtw_matrix[i - 1, j - 2]
            ))

            if tb == 0:
                i = i - 1
            elif tb == 1:
                i = i - 1
                j = j - 1
            elif tb == 2:
                i = i - 1
                j = j - 2

            p.insert(0, i - 1)
            q.insert(0, j - 1)

    elif slope_constraint == 'symmetric':
        while (i > 1 or j > 1):
            tb = np.argmin((
                dtw_matrix[i - 1, j - 1],
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1]
            ))

            if tb == 0:
                i = i - 1
                j = j - 1

            elif tb == 1:
                i = i - 1

            elif tb == 2:
                j = j -1

            p.insert(0, i - 1)
            q.insert(0, j - 1)

    else:
        raise ValueError(f'Unknown slope constraint {slope_constraint}')

    return (np.array(p), np.array(q))


def dtw(
    array_1: np.ndarray,
    array_2: np.ndarray,
    return_flag: int = RETURN_VALUE,
    slope_constraint: str = 'asymmetric',
    window = None
):
    """_summary_

    Args:
        array_1 (np.ndarray): _description_
        array_2 (np.ndarray): _description_
        return_flag (int, optional): _description_. Defaults to RETURN_VALUE.
        slope_constraint (str, optional): _description_. Defaults
            to 'asymmetric'.
        window (_type_, optional): _description_. Defaults to None.
    """
    p = array_1.shape[0]
    assert p != 0, 'Array_1 is empty!'
    s = array_2.shape[0]
    assert s != 0, 'Array_2 is empty!'

    if window is None:
        window = s

    cost = np.full((p, s), np.inf)
    for i in range(p):
        start = max(0, i - window)
        end = min(s, i + window) + 1
        cost[i, start: end] = np.linalg.norm(
            array_2[start: end] - array_1[i],
            axis=1
        )

    dtw_ = _cummulative_matrix(cost, slope_constraint, window)

    if return_flag == RETURN_ALL:
        return (
            dtw_[-1, -1],
            cost,
            dtw_[1:, 1:],
            _traceback(dtw_, slope_constraint)
        )

    elif return_flag == RETURN_PATH:
        return _traceback(dtw_, slope_constraint)

    else:
        return dtw_[-1, -1]
