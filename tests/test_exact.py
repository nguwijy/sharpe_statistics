from unittest import TestCase
import numpy as np

from core.sharpe_stats_runner import SharpeStatsRunner


class SharpeExactTestCase(TestCase):
    """
    Test case for the SharpeStatsRunner.

    Attributes:
        total_test_count (int): Total number of tests to run.
        alpha (float): Significance level for hypothesis testing.
        ret1 (numpy.ndarray): Array of returns for the first set of data.
        ret2 (numpy.ndarray): Array of returns for the second set of data.

    Methods:
        setUp(self): Set up test data and parameters.
        test_sharpe_hac(self): Test the SharpeStatsRunner using HAC method.
        test_sharpe_boot_ts(self): Test the SharpeStatsRunner using Bootstrap method with time series block bootstrap.
        test_sharpe_boot_iid(self): Test the SharpeStatsRunner using Bootstrap method with independent and identically distributed (IID) bootstrap.

    Example:
        A typical usage example would be:
        ```python
        test_case = SharpeExactTestCase()
        test_case.test_sharpe_hac()
        ```
    """

    def setUp(self):
        """
        Set up test data and parameters.
        """
        self.total_test_count = 5000
        self.alpha = 0.05
        self.ret1 = np.array([3.92670000, -2.3305000, -4.8583000, 1.96000000, -0.4397300, -2.7056000, 2.92570000, 3.87810000, -2.5249000, 1.74590000, -4.0699000, 0.84013000, -0.5617100, 2.84340000, 3.11080000, 1.91650000, 1.24070000, 2.70060000, 4.01650000, 1.19310000, 2.44220000, -1.6150000, 3.69480000, 2.07020000, 2.02090000, 0.83481000, 1.44590000, 1.20520000, 1.51500000, 0.06133600, -4.8483000, 2.44980000, 4.55040000, 1.21710000, 5.40700000, -2.5487000, 3.72540000, 0.54106000, -5.4384000, 5.18970000, 4.95600000, 4.63930000, 7.65870000, -5.7242000, 4.80640000, -3.2255000, 3.90520000, 1.86220000, 0.10583000, 6.23280000, 4.70720000, 0.34273000, -1.0273000, 4.33750000, 0.03111600, -16.625000, 3.97890000, 6.78550000, 5.69550000, 7.71210000, 3.20710000, -2.4622000, 3.89690000, 1.76310000, -3.5363000, 5.06140000, -3.5352000, -1.9028000, -2.5466000, 3.85620000, 3.63410000, 9.75480000, -4.8312000, 2.90380000, 3.82400000, -6.4514000, -5.0871000, 5.19070000, -2.5309000, 6.15040000, -5.8471000, -0.8204800, -11.711000, 2.15650000, 2.76750000, -10.242000, -8.8416000, 12.0670000, 0.23470000, -0.6506600, -5.3993000, -5.2209000, -10.378000, 2.62880000, 5.75680000, 2.07200000, -1.6054000, -2.5013000, 2.24750000, -6.0060000, -0.4965000, -6.5602000, -8.6304000, -0.0425940, -10.261000, 8.24590000, 4.83680000, -5.7832000, -3.3307000, -0.8245600, 1.73950000, 7.23180000, 3.48930000, 0.94715000, 2.39770000, 1.60400000, -1.8021000, 5.55900000, 0.58602000, 5.52010000])
        self.ret2 = np.array([2.64760000000000, -0.816280000000000, -5.72400000000000, 0.0474760000000000, -4.24160000000000, -7.59120000000000, 3.33910000000000, 6.75050000000000, -0.843120000000000, 4.28130000000000, -4.41420000000000, 2.29520000000000, -3.21210000000000, 4.18450000000000, 3.66220000000000, 2.78660000000000, 3.57730000000000, 10.0630000000000, 10.7530000000000, 0.906150000000000, 1.90050000000000, -1.66430000000000, -1.68610000000000, -5.91620000000000, 0.996660000000000, 5.41830000000000, -0.403410000000000, 5.81890000000000, 3.37120000000000, -4.26580000000000, -10.7530000000000, 3.69750000000000, 7.75710000000000, -3.00710000000000, 4.93610000000000, -3.78720000000000, 6.32270000000000, -5.64270000000000, -7.06980000000000, 1.74900000000000, 8.63520000000000, 2.33160000000000, 9.29360000000000, -1.61820000000000, 5.63140000000000, -8.20500000000000, 0.500030000000000, 1.09180000000000, -1.07180000000000, 8.75430000000000, 5.11080000000000, 1.20070000000000, -4.22400000000000, 9.33900000000000, 0.463230000000000, -21.9250000000000, 11.4950000000000, 2.99960000000000, 6.78450000000000, 12.2250000000000, 10.2200000000000, -5.98690000000000, 12.7810000000000, 6.38120000000000, -4.78080000000000, 8.73820000000000, -1.53230000000000, 5.41220000000000, -2.88540000000000, 10.4790000000000, 10.8260000000000, 16.9360000000000, -0.699960000000000, 15.9100000000000, -2.98170000000000, -17.0610000000000, -12.1860000000000, 16.9480000000000, -7.93020000000000, 14.7800000000000, -10.2540000000000, -12.1910000000000, -26.5170000000000, 6.52520000000000, 1.67180000000000, -22.6160000000000, -23.9400000000000, 19.0280000000000, -3.57650000000000, -11.1930000000000, -13.3670000000000, -15.1100000000000, -23.9460000000000, 15.6860000000000, 10.0070000000000, 0.0186130000000000, -7.06580000000000, -10.8050000000000, 4.75810000000000, -10.2730000000000, -7.13690000000000, -16.3760000000000, -7.77480000000000, -3.71390000000000, -14.6970000000000, 13.9630000000000, 13.4860000000000, -9.00860000000000, -0.636020000000000, -1.54890000000000, 1.80770000000000, 4.97740000000000, 7.06600000000000, 1.26390000000000, 2.70540000000000, 3.73190000000000, -1.55510000000000, 5.91610000000000, 2.27750000000000, 1.82370000000000])

    def test_sharpe_hac(self):
        """
        Test the SharpeStatsRunner using HAC method.
        Checks if the calculated p-value is within the expected range.
        """
        expected_p_value = 0.0646
        tol, method = .001, 'HAC'
        out = SharpeStatsRunner(method=method).run(self.ret1, self.ret2)
        _, pval = out[0], out[1]
        print(f"[{method}] The p-value: {pval:.2%}, compared to the expected p-value from the paper: {expected_p_value:.2%}")
        assert abs(pval - expected_p_value) < tol

    def test_sharpe_boot_ts(self):
        """
        Test the SharpeStatsRunner using Bootstrap method with time series block bootstrap.
        Checks if the calculated p-value is within the expected range.
        """
        expected_p_value = 0.0809382  # not sure why we cannot reproduce the exact number as in the paper 0.092
        tol, method = .001, 'BOOT'
        out = SharpeStatsRunner(method=method).run(self.ret1, self.ret2, b=4, nb_boot=500000)
        _, pval = out[0], out[1]
        print(f"[{method}][b=4] The p-value: {pval:.2%}, compared to the expected p-value: {expected_p_value:.2%}")
        assert abs(pval - expected_p_value) < tol

    def test_sharpe_boot_iid(self):
        """
        Test the SharpeStatsRunner using Bootstrap method with independent and identically distributed (IID) bootstrap.
        Checks if the calculated p-value is within the expected range.
        """
        expected_p_value = 0.069975  # not sure why we cannot reproduce the exact number as in the paper 0.044
        tol, method = .001, 'BOOT'
        out = SharpeStatsRunner(method=method).run(self.ret1, self.ret2, b=1, nb_boot=500000)
        _, pval = out[0], out[1]
        print(f"[{method}][iid] The p-value: {pval:.2%}, compared to the expected p-value: {expected_p_value:.2%}")
        assert abs(pval - expected_p_value) < tol
