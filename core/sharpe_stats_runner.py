import numpy as np

from .sharpe_bootstrap import sharpe_boot
from .sharpe_hac import sharpe_hac


class SharpeStatsRunner:
    def __init__(self, method: str = 'HAC'):
        """
        Initializes the SharpeStats instance.

        Parameters:
        - method (str): The method to be used for Sharpe test. Options: 'standard' or 'hac'.
                       Defaults to 'standard'.
        """
        self.method = method

    def run(self, ret1: np.ndarray, ret2: np.ndarray, **kwargs) -> tuple:
        """
        Runs the Sharpe test based on the specified method.

        Parameters:
        - ret1, ret2 (np.ndarray): Arrays of returns for two sequences.

        Returns:
        tuple: Tuple containing (diff, pval) where:
            - diff (float): The difference in Sharpe (sharpe1 - sharpe2).
            - pval (float): P-value under the assumption that H0 holds.
                           H0: there is no difference between these two Sharpe ratios.

        For mathematical principles, please refer to the paper: www.ledoit.net/jef_2008pdf.pdf
        """
        if self.method == 'HAC':
            return sharpe_hac(ret1, ret2, **kwargs)
        elif self.method == 'BOOT':
            return sharpe_boot(ret1, ret2, **kwargs)
        else:
            raise ValueError("Invalid method. Supported methods: 'HAC'.")
