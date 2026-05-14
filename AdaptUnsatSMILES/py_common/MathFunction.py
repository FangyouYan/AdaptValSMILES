import copy

import numpy as np
from numpy.linalg import norm

def DicDelete(iD, SI):
    """
    delete values by index from dictionary
    :param iD: the index for deleting
    :param SI: a dictionary
    :return: SI(a dictionary after deletion)
    """
    SI = copy.deepcopy(SI)
    L = len(iD)
    for i, j in SI.items():
        if j.shape[0] > L:
            j = np.delete(j, iD, 0)
        if j.shape[1] > L:
            j = np.delete(j, iD, 1)
        SI[i] = j
    return SI


def norm_d(M, norm_k):
    """
    The deinition of norm for matrix M
    :param M: matrix M
    :param norm_k: norm id
    :return:norm_(norm value)
    """
    if norm_k == 1:
        norm_ = np.mean(np.sum(abs(M), axis=0))
    elif norm_k == 2:
        norm_ = abs(M).sum(axis=0).sum(axis=0)
    elif norm_k == 3:
        norm_ = np.sqrt(np.max(np.sum(np.power(M, 2), axis=0)))
    elif norm_k == 4:
        norm_ = norm(M, ord='fro')
    elif norm_k == 5:
        norm_ = norm(M, ord=2)
    elif norm_k == 6:
        norm_ = norm(M, ord=1)
    else:
        raise ValueError("norm_k must be between 1 and 6")
    return norm_




def cal_std_95(Y):
    """
    Calculate the standard deviation of values within the 95% confidence interval.
    :param Y: array-like, 1D array
    :return: std_95, standard deviation of values within the 95% confidence interval.
    """
    mean = np.mean(Y)
    std = np.std(Y)
    lower_limit = mean - 1.96 * std
    upper_limit = mean + 1.96 * std
    sample_95 = Y[(Y>=lower_limit) & (Y<=upper_limit)]
    std_95 = np.std(sample_95)

    return std_95



def weighted_average(extra_degree_all, se_i=None, main_weight=None):
    eda_arr = np.array(extra_degree_all, dtype=float)
    n = len(eda_arr)

    if main_weight is None:
        return eda_arr.mean()

    other_weight_sum = 1 - main_weight
    other_weight = other_weight_sum / (n - 1)

    w = np.full(n, other_weight)
    w[se_i] = main_weight

    return np.sum(eda_arr * w)

def extrapolation_degree(x_train, x_test, se_i=None, main_weight=None):
    """
    Calculate the extrapolation degree.
    :param x_train: training set
    :param x_test: test set
    :param se_i: index of ii
    :param main_weight: main weight
    :return:extra_degree,extrapolation degree
    """
    extra_degree_all = []
    x_train_max = np.max(x_train[:, se_i])
    x_train_min = np.min(x_train[:, se_i])

    if np.any(x_test[:, se_i] > x_train_max):
        extrap_direction = 'forward'
    elif np.any(x_test[:, se_i] < x_train_min):
        extrap_direction = 'backward'
    else:
        extrap_direction = 'interpolation'

    if (extrap_direction=='forward') or (extrap_direction=='backward'):
        for j in range(0, int(np.shape(x_train)[1])):
            x_train_max = np.max(x_train[:, j])
            x_train_min = np.min(x_train[:, j])
            x_train_mean = np.mean(x_train[:, j])

            x_test_min = x_train_min - x_test[:, j]
            x_test_min[np.where(x_test_min < 0)] = 0
            x_test_max = x_test[:, j] - x_train_max
            x_test_max[np.where(x_test_max < 0)] = 0
            x_test_gap = abs(x_train_mean - x_test[:, j])
            x_test_gap[np.where((x_test_min == 0) & (x_test_max == 0))] = 0

            if np.sum(x_test_gap) == 0:
                extra_degree_x = 0
            else:
                extra_degree_x = (np.sum(x_test_min) + np.sum(x_test_max)) / np.sum(x_test_gap)

            extra_degree_all.append(extra_degree_x)

        extra_degree = weighted_average(extra_degree_all, se_i=se_i, main_weight=main_weight)
    else:
        extra_degree = 0

    return extra_degree,extrap_direction



