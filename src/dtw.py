from tslearn.utils import to_time_series
import numpy as np
from numba import njit, prange

def dtw(s1, s2, radius, mean_norm=False):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return both the path and the
    similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} (X_{i} - Y_{j})^2}

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1
        A time series.

    s2
        Another time series.

    radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2

    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_path([1, 2, 3], [1., 2., 2., 3.])
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> dist
    0.0
    >>> dtw_path([1, 2, 3], [1., 2., 2., 3., 4.])[1]
    1.0

    See Also
    --------
    dtw : Get only the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets
    dtw_path_from_metric : Compute a DTW using a user-defined distance metric

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    if (None in s1) or (None in s2):
        raise Exception('Input arrays must not contain None values')

    radius = 2 if radius < 2 else radius
    
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    if len(s1) == 0 or len(s2) == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length.")

    if s1.shape[1] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")

    if mean_norm:
        s1 = (s1-s1.mean(axis=0)) #/s1.std()
        s2 = (s2-s2.mean(axis=0)) #/s2.std()

    sz1 = s1.shape[0]
    sz2 = s2.shape[0]

    mask = sc_mask(sz1, sz2, radius=radius)

    acc_cost_mat = njit_accumulated_matrix(s1, s2, mask=mask)

    #(sx, sy), (ex, ey) =  _start_end_ix(s1, s2, radius)

    #acc_cost_cut = acc_cost_mat[sx:ex+1,sy:ey+1]

    #path = _return_path(acc_cost_cut)
    path = _return_path(acc_cost_mat)

    #path = np.array([[x+sx, y+sy] for x,y in path])

    dtw_val = path_to_distance(s1, s2, path)

    return path, dtw_val


def line(x0, y0, x1, y1):
    "Bresenham's line algorithm"
    points_in_line = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points_in_line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points_in_line.append((x, y))
    return np.array(points_in_line)


def sc_mask(sz1, sz2, radius=1):
    """Compute the Sakoe-Chiba mask.

    Parameters
    ----------
    sz1 : int
        The size of the first time series

    sz2 : int
        The size of the second time series.

    radius : int
        The radius of the band

    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Sakoe-Chiba mask.
    """
    mask = np.full((sz1, sz2), 0.0)
    
    if radius >= sz1 or radius >= sz1:
        return mask

    #for i in range(-radius, radius):

    #start = (-i, 0) if i<0 else (0, i)
    #end = (sz2-1, sz1-1-i) if i<0 else (sz2-1-i, sz1-1)
    #points_in_line = line(start[0], start[1], end[0], end[1])

    #points_in_line = [(x,y) for x,y in points_in_line if x<sz1 and y<sz2]
    #mask[points_in_line] = 1
    start_low = (radius, 0)
    end_low = (sz1-1, sz2-1-radius)
    
    start_high = (0, radius)
    end_high =  (sz1-1-radius, sz2-1)
    
    low_points = line(start_low[0], start_low[1], end_low[0], end_low[1])
    high_points = line(start_high[0], start_high[1], end_high[0], end_high[1])
    
    x_low, y_low = low_points.T
    x_high, y_high = high_points.T
    
    # fill low
    for x,y in low_points:
        mask[x:,y] = np.Inf
    
    for x,y in high_points:
        mask[x,y:] = np.Inf

    return mask
 

@njit()
def path_to_distance(s1, s2, path):
    distances = np.abs(s1[path[:, 0]] - s2[path[:, 1]]) ** 2
    dtw_val = np.sqrt(np.sum(distances))
    return dtw_val


def _start_end_ix(s1, s2, r):

    #if r >= len(s1)/2 or r >= len(s2)/2:
        #r = int(max([len(s1), len(s2)])/2)

    s1l = len(s1)
    s2l = len(s2)

    starting_zone = njit_dist_matrix(s1[:r], s2[:r])
    ending_zone = njit_dist_matrix(s1[-r:], s2[-r:])
    
    # automatically takes earliest index (top left of array)
    sz1, sz2 = np.unravel_index(np.argmin(starting_zone, axis=None), starting_zone.shape)

    # Ensure we get latest index (botom right of array)
    exmin, eymin = np.where(ending_zone==np.min(ending_zone))
    ez1, ez2 = exmin[-1], eymin[-1]

    # Convert to indices in original arrays
    s1f = s1l-r+ez1
    s2f = s2l-r+ez2

    return (sz1, sz2), (s1f, s2f)


@njit()
def _return_path(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = np.array([acc_cost_mat[i - 1][j - 1],
                               acc_cost_mat[i - 1][j],
                               acc_cost_mat[i][j - 1]])
            argmin = np.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return np.array(path[::-1])


@njit()
def _local_squared_dist(x, y):
    dist = 0.
    for di in range(x.shape[0]):
        diff = (x[di] - y[di])
        dist += diff * diff
    return dist

@njit()
def njit_accumulated_matrix(s1, s2, mask):
    """Compute the accumulated cost matrix score between two time series.

    Parameters
    ----------
    s1 : array, shape = (sz1,)
        First time series.

    s2 : array, shape = (sz2,)
        Second time series

    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    mat : array, shape = (sz1, sz2)
        Accumulated cost matrix.

    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            if np.isfinite(mask[i, j]):
                cum_sum[i + 1, j + 1] = _local_squared_dist(s1[i], s2[j])
                cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                             cum_sum[i + 1, j],
                                             cum_sum[i, j])
    return cum_sum[1:, 1:]


@njit()
def njit_dist_matrix(s1, s2):
    """Compute the cost matrix score between two time series.

    Parameters
    ----------
    s1 : array, shape = (sz1,)
        First time series.

    s2 : array, shape = (sz2,)
        Second time series

    Returns
    -------
    mat : array, shape = (sz1, sz2)
        cost matrix.

    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    
    dists = np.zeros((l1, l2))

    for i in range(l1):
        for j in range(l2):
            sd = _local_squared_dist(s1[i], s2[j])
            dists[i, j] = sd

    return dists

