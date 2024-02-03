from typing import Tuple
import numpy as np

from core.sharpe_hac import sharpe_ratio_diff, _compute_gradient, sharpe_hac, _compute_vhat


def block_size_calibrate(ret: np.ndarray, b_vec: np.ndarray = np.array([1, 2, 4, 6, 8, 10]).reshape(-1, 1),
                         alpha: float = 0.05, nb_inner_boot: int = 199, nb_outer_boot: int = 2000,
                         b_av: int = 5, seq_start: int = 50, se_type: str = 'G') -> Tuple[int, np.ndarray]:
    """
    Calibrate the block size for the Sharpe test.

    Args:
        ret (np.ndarray): T*2 matrix of returns.
        b_vec (np.ndarray, optional): Vector of candidate block sizes. Defaults to [1, 2, 4, 6, 8, 10].
        alpha (float, optional): Nominal significance level. Defaults to 0.05.
        nb_inner_boot (int, optional): Number of 'inner' bootstrap repetitions. Defaults to 199.
        nb_outer_boot (int, optional): Number of 'outer' bootstrap repetitions. Defaults to 2000.
        b_av (int, optional): Average block size for stationary bootstrap. Defaults to 5.
        seq_start (int, optional): Number of 'warm up' observations for VAR generated data. Defaults to 50.
        se_type (str, optional): Type of standard error for 'original' test statistic. Defaults to 'G'.

    Returns:
        Tuple[int, np.ndarray]: Optimal block size and array of candidate block sizes
        with corresponding simulated rejection probabilities.
    """

    b_len = len(b_vec)
    emp_reject_probs = np.zeros(b_len)
    ret1 = ret[:, 0]
    ret2 = ret[:, 1]
    diff_hat = sharpe_ratio_diff(ret1, ret2)
    seq_len = len(ret1)
    var_data = np.zeros((seq_start + seq_len, 2))
    var_data[0, :] = ret[0, :]
    y1 = ret1[1:seq_len]
    y2 = ret2[1:seq_len]
    x1 = ret1[0:(seq_len - 1)]
    x2 = ret2[0:(seq_len - 1)]

    X = np.column_stack([np.ones(seq_len - 1), x1, x2])
    coef1 = np.linalg.lstsq(X.T @ X, X.T @ y1)[0]
    coef2 = np.linalg.lstsq(X.T @ X, X.T @ y2)[0]
    resid1 = y1 - X @ coef1
    resid2 = y2 - X @ coef2
    resid_mat = np.column_stack([resid1, resid2])

    for _ in range(nb_outer_boot):
        resid_mat_star = np.vstack([np.zeros(2), resid_mat[sb_sequence(seq_len - 1, b_av, seq_start + seq_len - 1), :]])

        for t in range(1, seq_start + seq_len):
            var_data[t, 0] = coef1[0] + coef1[1] * var_data[t - 1, 0] + coef1[2] * var_data[t - 1, 1] + resid_mat_star[t, 0]
            var_data[t, 1] = coef2[0] + coef2[1] * var_data[t - 1, 1] + coef2[2] * var_data[t - 1, 1] + resid_mat_star[t, 1]

        var_data_trunc_1 = var_data[seq_start:(seq_start + seq_len), 0]
        var_data_trunc_2 = var_data[seq_start:(seq_start + seq_len), 1]

        for j in range(b_len):
            p_value, delta_hat_star, _, _ = sharpe_boot(var_data_trunc_1, var_data_trunc_2, int(b_vec[j]), nb_inner_boot, se_type, diff_hat)

            if p_value <= alpha:
                emp_reject_probs[j] += 1

    emp_reject_probs /= nb_outer_boot
    b_sort = np.argsort(np.abs(emp_reject_probs - alpha))
    b_opt = int(b_vec[b_sort[0]])

    b_vec_with_probs = np.column_stack([b_vec.flatten(), emp_reject_probs])
    return b_opt, b_vec_with_probs


def cbb_sequence(seq_len: int, b: int) -> np.ndarray:
    """
    Computes a circular block bootstrap sequence applied to [0:T-1].

    Args:
        seq_len (int): Length of the sequence.
        b (int): Block size.

    Returns:
        np.ndarray: Bootstrap sequence.
    """
    ll = int(np.floor(seq_len / b) + 1)
    index_sequence = np.concatenate([np.arange(seq_len), np.arange(b)])
    sequence = np.zeros(seq_len + b)

    start_points = np.random.randint(1, seq_len + 1, size=ll)

    for j in range(1, ll + 1):
        start = start_points[j - 1]
        sequence[((j - 1) * b):(j * b)] = index_sequence[(start - 1):(start + b - 1)]

    sequence = sequence[:seq_len]
    return sequence.astype(int)


def sb_sequence(seq_len: int, b_av: int, length: int = None) -> np.ndarray:
    """
    Computes a stationary bootstrap sequence applied to [0:T-1].

    Args:
        seq_len (int): Length of the sequence.
        b_av (int): Average block size.
        length (int, optional): Length of bootstrap sequence. Defaults to None.

    Returns:
        np.ndarray: Bootstrap sequence.
    """
    if length is None:
        length = seq_len

    index_sequence = np.concatenate([np.arange(seq_len), np.arange(seq_len)])
    sequence = np.zeros(length + seq_len)

    current = 0

    while current < length:
        start = np.random.permutation(seq_len)[0]
        b = np.random.geometric(1 / b_av) + 1
        sequence[current:(current + b)] = index_sequence[start:(start + b)]
        current += b

    sequence = sequence[:length]
    return sequence.astype(int)


def sharpe_boot(ret1: np.ndarray, ret2: np.ndarray, b: int = None, nb_boot: int = 4999,
                se_type: str = 'G', diff_null: float = 0, **kwargs) -> Tuple[float, float, float, int]:
    """
    Performs Sharpe test with bootstrapping.

    Args:
        ret1 (np.ndarray): Returns for the first asset.
        ret2 (np.ndarray): Returns for the second asset.
        b (int, optional): Block size. Defaults to None.
        nb_boot (int, optional): Number of bootstrap repetitions. Defaults to 4999.
        se_type (str, optional): Type of standard error. Defaults to 'G'.
        diff_null (float, optional): Null hypothesis difference. Defaults to 0.

    Returns:
        Tuple[float, float, float, int]: Tuple containing difference, p-value, test statistic, and block size.
    """
    if b is None:
        b = block_size_calibrate(np.stack([ret1, ret2], axis=1))

    # Compute observed difference in Sharpe ratios
    diff = sharpe_ratio_diff(ret1, ret2)

    # Compute HAC standard error
    _, _, se = sharpe_hac(ret1, ret2)

    # Compute 'original' test statistic
    d = abs(diff - diff_null) / se

    b_root = np.sqrt(b)
    seq_len = len(ret1)
    ll = int(np.floor(seq_len / b))

    # Adjusted sample size for block bootstrap (using a multiple of the block size)
    t_adj = ll * b

    slicings = np.array([cbb_sequence(t_adj, b) for _ in range(nb_boot)])
    ret1_star, ret2_star = ret1[slicings], ret2[slicings]
    diff_star = sharpe_ratio_diff(ret1_star, ret2_star)
    y_star, ret_stats = _compute_vhat(ret1_star, ret2_star)
    gradient = _compute_gradient(*ret_stats)

    # Compute bootstrap standard error
    zeta_star = (b_root * y_star.reshape(nb_boot, ll, b, -1).mean(axis=2))
    psi_hat_star = (zeta_star[:, :, :, None] * zeta_star[:, :, None, :]).mean(axis=1)
    se_star = np.sqrt(np.matmul(gradient.transpose(0, 2, 1), np.matmul(psi_hat_star, gradient)) / t_adj).reshape(-1)

    # Compute bootstrap test statistic (and update p-value accordingly)
    d_star = abs(diff_star - diff) / se_star
    p_value = (1 + (d_star >= d).sum()) / (nb_boot + 1)

    return diff, p_value, d, b
