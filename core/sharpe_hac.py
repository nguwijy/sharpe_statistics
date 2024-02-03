import numpy as np
import scipy.integrate as integrate
from typing import Callable, Tuple


def _compute_vhat(ret1: np.ndarray, ret2: np.ndarray) -> Tuple:
    """
    Compute the matrix of demeaned returns.

    Parameters:
    - ret1 (np.ndarray): First return sequence.
    - ret2 (np.ndarray): Second return sequence.

    Returns:
    - List: moments of returns
    - np.ndarray: Matrix of demeaned returns.
    """
    nu = [np.mean(ret1, axis=-1), np.mean(ret2, axis=-1), np.mean(ret1 ** 2, axis=-1), np.mean(ret2 ** 2, axis=-1)]
    nux = np.stack([ret1 - nu[0].reshape(-1, 1), ret2 - nu[1].reshape(-1, 1), ret1 ** 2 - nu[2].reshape(-1, 1), ret2 ** 2 - nu[3].reshape(-1, 1)], axis=-1)
    return nux, nu


def _ar2(y: np.ndarray, nlag: int, const: int = 1) -> dict:
    """
    Calculate OLS estimates for the AR(k) model.

    Parameters:
    - y (np.ndarray): Return sequence.
    - nlag (int): Number of lags in the AR model.
    - const (int, optional): Whether to include a constant term. Default is 1.

    Returns:
    - dict: Dictionary containing AR(2) model results.
    """
    results = {}
    n = y.shape[0]
    results['y'] = y
    results['negs'] = 1
    cols = []
    if const == 1:
        cols.append(np.ones(n))
    for i in range(nlag):
        cols.append(np.roll(y, i + 1))

    x = np.vstack(cols).T
    x = x[nlag:, :]
    y = y[nlag:]
    n_adj = len(y)

    b0 = np.linalg.lstsq(x.T.dot(x), x.T.dot(y), rcond=None)[0]
    p = nlag + const
    sige = ((y - x.dot(b0)).T.dot(y - x.dot(b0))) / (n - p + 1)

    results['meth'] = 'ar'
    results['beta'] = b0
    results['sige'] = sige
    results['yhat'] = x.dot(b0)
    results['nobs'] = n
    results['nadj'] = n_adj
    results['nvar'] = nlag * const
    results['x'] = x
    return results


def _compute_alpha(v_hat: np.ndarray) -> float:
    """
    Compute alpha parameter using AR(2) model results.

    Parameters:
    - v_hat (np.ndarray): Matrix of demeaned returns.

    Returns:
    - float: Computed alpha parameter.
    """
    t = v_hat.shape[0]
    p = v_hat.shape[1]
    numerator = 0
    denominator = 0
    for i in range(p):
        results = _ar2(v_hat[:, i], 1)
        rho_hat = results['beta'][1]
        sig_hat = np.sqrt(results['sige'])
        numerator = numerator + 4 * rho_hat ** 2 * sig_hat ** 4 / (1 - rho_hat) ** 8
        denominator = denominator + sig_hat ** 4 / (1 - rho_hat) ** 4

    alpha_hat = numerator / denominator
    return alpha_hat


def _gamma_hat(v_hat: np.ndarray, j: int) -> np.ndarray:
    """
    Compute gamma matrix using demeaned returns and lag parameter.

    Parameters:
    - v_hat (np.ndarray): Matrix of demeaned returns.
    - j (int): Lag parameter.

    Returns:
    - np.ndarray: Computed gamma matrix.
    """
    t = v_hat.shape[0]
    p = v_hat.shape[1]
    gamma_hat = np.zeros((p, p))
    if j >= t:
        raise Exception('j must be smaller than the row dimension!')
    else:
        for i in range(j, t):
            gamma_hat = gamma_hat + v_hat[i, :].reshape(-1, 1).dot(v_hat[i - j, :].reshape(1, -1))
        gamma_hat = gamma_hat / t
        return gamma_hat


def _kernel_type(x: float, kernel_type: str) -> float:
    """
    Compute kernel weight based on kernel type and input value.

    Parameters:
    - x (float): Input value.
    - kernel_type (str): Kernel type ('G' or 'QS').

    Returns:
    - float: Computed kernel weight.
    """
    if kernel_type == 'G':
        if x < 0.5:
            wt = 1 - 6 * x ** 2 + 6 * x ** 3
        elif x < 1:
            wt = 2 * (1 - x) ** 3
        else:
            wt = 0
        return wt
    elif kernel_type == 'QS':
        term = 6 * np.pi * x / 5
        wt = 25 * (np.sin(term) / term - np.cos(term)) / (12 * np.pi ** 2 * x ** 2)
        return wt
    else:
        raise Exception('wrong type')


def _compute_psi(v_hat: np.ndarray, kernel_type: str = 'G') -> np.ndarray:
    """
    Compute Psi matrix using demeaned returns.

    Parameters:
    - v_hat (np.ndarray): Matrix of demeaned returns.
    - kernel_type (str): Kernel type ('G' or 'QS').

    Returns:
    - np.ndarray: Computed Psi matrix.
    """
    t = v_hat.shape[0]
    alpha_hat = _compute_alpha(v_hat)
    ss_star = 2.6614 * (alpha_hat * t) ** 0.2
    psi_hat = _gamma_hat(v_hat, 0)
    j = 1
    while j < ss_star:
        gamma = _gamma_hat(v_hat, j)
        psi_hat = psi_hat + _kernel_type(j / ss_star, kernel_type) * (gamma + gamma.T)
        j += 1
    psi_hat = (t / (t - 4)) * psi_hat
    return psi_hat


def _compute_gradient(mu1_hat: float, mu2_hat: float, gamma1_hat: float, gamma2_hat: float) -> np.ndarray:
    """
    Compute gradient based on return statistics.

    Parameters:
    - mu1_hat (float): Mean of the first return sequence
    - mu2_hat (float): Mean of the second return sequence
    - gamma1_hat (float): Second moment of the first return sequence
    - gamma2_hat (float): Second moment of the second return sequence

    Returns:
    - np.ndarray: Computed gradient statistics.
    """
    return np.array([[
        gamma1_hat / (gamma1_hat - mu1_hat ** 2) ** 1.5,
        -gamma2_hat / (gamma2_hat - mu2_hat ** 2) ** 1.5,
        -0.5 * mu1_hat / (gamma1_hat - mu1_hat ** 2) ** 1.5,
        0.5 * mu2_hat / (gamma2_hat - mu2_hat ** 2) ** 1.5,
    ]]).T


def _compute_se(ret1: np.ndarray, ret2: np.ndarray) -> np.ndarray:
    """
    Compute standard error of the difference in Sharpe ratios.

    Parameters:
    - ret1 (np.ndarray): First return sequence.
    - ret2 (np.ndarray): Second return sequence.

    Returns:
    - np.ndarray: Computed standard error.
    """
    t = len(ret1)
    v_hat, ret_stats = _compute_vhat(ret1, ret2)
    gradient = _compute_gradient(*ret_stats)
    psi_hat = _compute_psi(v_hat.reshape(t, -1))
    se = np.sqrt(gradient.T.dot(psi_hat).dot(gradient) / t)
    return se


def sharpe_ratio_diff(ret1: np.ndarray, ret2: np.ndarray) -> float:
    """
    Computes the difference between two Sharpe ratios.

    Parameters:
    - ret1 (np.ndarray): batch x T matrix of returns.
    - ret2 (np.ndarray): batch x T matrix of returns.

    Returns:
    float: Difference of the two Sharpe ratios.
    """
    mu1_hat = np.mean(ret1, axis=-1)
    mu2_hat = np.mean(ret2, axis=-1)
    sig1_hat = np.std(ret1, axis=-1)
    sig2_hat = np.std(ret2, axis=-1)

    sr1_hat = mu1_hat / sig1_hat
    sr2_hat = mu2_hat / sig2_hat

    diff = sr1_hat - sr2_hat
    return diff


def sharpe_hac(ret1: np.ndarray, ret2: np.ndarray, **kwargs) -> tuple:
    """
    Perform Sharpe ratio test on two return sequences.

    Parameters:
    - ret1 (np.ndarray): First return sequence.
    - ret2 (np.ndarray): Second return sequence.

    Returns:
    - tuple: Difference in Sharpe ratios and p-value.
    """
    diff = sharpe_ratio_diff(ret1, ret2)
    se = _compute_se(ret1, ret2)[0, 0]
    func: Callable = lambda x: (1 / np.sqrt(np.pi * 2)) * np.exp(-0.5 * x ** 2)
    pval, err = integrate.quad(func, -1000, -abs(diff) / se)
    pval = 2 * pval
    return diff, pval, se
