"""

"""
import os

import numpy as np
import scipy
import scipy.signal as sig
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.0f"


y_scalar_formatter = ScalarFormatterClass(useMathText=True)
y_scalar_formatter.set_powerlimits((0, 0))
plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["figure.autolayout"] = True


def main(filepath_spikes, dir_output):

    # =========================================================================
    # PCA FROM LAB 5
    # =========================================================================

    # unpack data
    dict_spikes = scipy.io.loadmat(filepath_spikes)
    np_spikes = dict_spikes["spikes"]

    # normalize spikes
    np_spikes_mean = np.mean(np_spikes, axis=0, keepdims=True)
    np_spikes_std = np.std(np_spikes, axis=0, keepdims=True)
    np_spikes_normalized = (np_spikes - np_spikes_mean) / np_spikes_std

    # PCA
    n_dims_original = np_spikes_normalized.shape[1]
    np_unitary, np_singular_values, np_row_eigenvectors = np.linalg.svd(np_spikes_normalized, full_matrices=False)
    np_eigenvalues = np.square(np_singular_values) / (n_dims_original - 1)
    np_col_pcs = np_unitary @ np.diag(np_singular_values)

    # =========================================================================
    # PART 1: K-MEANS CLUSTERING
    # =========================================================================

    


if __name__ == "__main__":
    _filepath_spikes = r"D:\Documents\Academics\BME517\bme_lab_6\data\spikes.mat"
    _dir_output = r"D:\Documents\Academics\BME517\bme_lab_5_6_report"
    main(_filepath_spikes, _dir_output)
