import numpy as np
from scipy.optimize import linear_sum_assignment


def get_km_scores(times, labels, fail_code, sort=False):
    """
    # estimate KM survival rate
    :param times: ndarray, shape(num_subject, ), event times or censoring times, shape,
    :param labels: ndarray, shape(num_subject, ), event labels
    :param fail_code: event_id
    :param sort: whether sort by times, default False (we assume that the time is sorted in ascending order)
    :return:
    """
    N = len(times)
    times = np.reshape(times, [-1])
    labels = np.reshape(labels, [-1])
    # Sorting T and E in ascending order by T
    if sort:
        order = np.argsort(times)
        T = times[order]
        E = labels[order]
    else:
        T = times
        E = labels
    max_T = int(np.max(T)) + 1

    # calculate KM survival rate at time 0-T_max
    km_scores = np.ones(max_T)
    n_fail = 0
    n_rep = 0

    for i in range(N):

        if E[i] == fail_code:
            n_fail += 1

        if i < N - 1 and T[i] == T[i + 1]:
            n_rep += 1
            continue

        km_scores[int(T[i])] = 1. - n_fail / (N - i + n_rep)
        n_fail = 0
        n_rep = 0

    for i in range(1, max_T):
        km_scores[i] = km_scores[i - 1] * km_scores[i]

    return km_scores


def get_bh_mask(time, label, num_times):
    # Tensor to numpy
    time = time.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    N = len(time)
    T = np.reshape(time, [N])
    E = np.reshape(label, [N])
    mask = np.zeros([N, num_times])

    for i in range(N):
        if E[i] > 0:
            mask[i, int(T[i])] = 1
        else:
            mask[i, int(T[i] + 1):] = 1
    return mask


def cluster_acc(pred, label):
    D = max(pred.max(), label.max()) + 1
    w = np.zeros([D, D], dtype=np.int32)
    for i in range(pred.size):
        w[pred[i], label[i]] += 1
    row_ind, col_ind = linear_sum_assignment(np.max(w) - w)

    return w[row_ind, col_ind].sum() * 1. / pred.size, w
