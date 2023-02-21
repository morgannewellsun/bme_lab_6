"""

"""
import os

import numpy as np
import scipy
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.0f"


y_scalar_formatter = ScalarFormatterClass(useMathText=True)
y_scalar_formatter.set_powerlimits((0, 0))
plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["figure.autolayout"] = True


def distances_euclidian_squared(points_a, points_b):
    if len(points_a.shape) != 2:
        raise ValueError("points_a must have shape (n_points, n_dims).")
    if len(points_b.shape) != 2:
        raise ValueError("points_b must have shape (n_points, n_dims).")
    if points_a.shape[1] != points_b.shape[1]:
        raise ValueError("points_a and points_b must have the same n_dims.")
    np_points_a_b_displacements = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    np_points_a_b_euclidian_squared = np.sum(np.square(np_points_a_b_displacements), axis=2)
    return np_points_a_b_euclidian_squared


def distances_mahalanobis_squared(points, clusters_mean, clusters_covar_inv):
    if len(points.shape) != 2:
        raise ValueError("points must have shape (n_points, n_dims).")
    if len(clusters_mean.shape) != 2:
        raise ValueError("clusters_mean must have shape (n_clusters, n_dims).")
    if (len(clusters_covar_inv.shape) != 3) or (clusters_covar_inv.shape[1] != clusters_covar_inv.shape[2]):
        raise ValueError("clusters_covar_inv must have shape (n_clusters, n_dims, n_dims).")
    if clusters_mean.shape[0] != clusters_covar_inv.shape[0]:
        raise ValueError("clusters_mean, and clusters_covar_inv must have the same n_clusters.")
    if not (points.shape[1] == clusters_mean.shape[1] == clusters_covar_inv.shape[1]):
        raise ValueError("points, clusters_mean, and clusters_covar_inv must have the same n_dims.")
    np_clusters_points_displacements = points[np.newaxis, :, :] - clusters_mean[:, np.newaxis, :]
    foo = np.einsum("ijk,ikl->ijl", np_clusters_points_displacements, clusters_covar_inv)
    np_clusters_points_mahalanobis_squared = np.einsum("ijk,ijk->ij", foo, np_clusters_points_displacements)
    return np_clusters_points_mahalanobis_squared.T


class KMeans:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.n_data = None
        self.n_dims = None
        self.center_method = None
        self.distance_method = None
        self.np_data = None
        self.np_centroids = None
        self.np_centroids_history = None
        self.np_cluster_indices = None
        self.rng = np.random.default_rng()

    def fit(self, data, center_method, distance_method, convergence_threshold=0.):
        if center_method not in ("centroid", "medoid"):
            raise ValueError('center_method argument must be "centroid" or "medoid".')
        if distance_method not in ("euclidian", "mahalanobis"):
            raise ValueError('distance_method argument must be "euclidian" or "mahalanobis".')
        self.center_method = center_method
        self.distance_method = distance_method
        self.np_data = data
        self.n_data, self.n_dims = data.shape
        np_centroid_indices = self.rng.choice(self.n_data, size=self.n_clusters, replace=False)
        self.np_centroids = self.np_data[np_centroid_indices]
        self.np_centroids_history = [self.np_centroids]
        max_center_displacement = np.inf
        while max_center_displacement > convergence_threshold:
            max_center_displacement = self._fit_step_euclidian()
        if self.distance_method == "euclidian":
            return self.np_cluster_indices
        max_center_displacement = np.inf
        while max_center_displacement > convergence_threshold:
            max_center_displacement = self._fit_step_mahalanobis()
        return self.np_cluster_indices

    def _fit_step_euclidian(self):
        np_distances = distances_euclidian_squared(self.np_data, self.np_centroids)

        self.np_cluster_indices = np.argmin(np_distances, axis=1)
        if self.center_method == "centroid":
            fn_center = np.mean
        else:  # if self.center_method == "medoid":
            fn_center = np.median
        np_centroids_old = self.np_centroids
        self.np_centroids = np.array([
            fn_center(self.np_data[self.np_cluster_indices == i, :],axis=0)
            for i
            in range(self.n_clusters)])
        self.np_centroids_history.append(self.np_centroids)
        max_center_displacement = np.max(np.sum(np.square(self.np_centroids - np_centroids_old), axis=1))
        return max_center_displacement

    def _fit_step_mahalanobis(self):
        covar_inv_matrices = []
        for cluster_idx in range(self.n_clusters):
            covar_inv_matrices.append(np.linalg.inv(np.cov(self.np_data[self.np_cluster_indices == cluster_idx, :].T)))
        np_distances = distances_mahalanobis_squared(
            points=self.np_data, clusters_mean=self.np_centroids, clusters_covar_inv=np.array(covar_inv_matrices))

        self.np_cluster_indices = np.argmin(np_distances, axis=1)
        if self.center_method == "centroid":
            fn_center = np.mean
        else:  # if self.center_method == "medoid":
            fn_center = np.median
        np_centroids_old = self.np_centroids
        self.np_centroids = np.array([
            fn_center(self.np_data[self.np_cluster_indices == i, :], axis=0)
            for i
            in range(self.n_clusters)])
        self.np_centroids_history.append(self.np_centroids)
        max_center_displacement = np.max(np.sum(np.square(self.np_centroids - np_centroids_old), axis=1))
        return max_center_displacement


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

    # keep only first two principal components
    np_col_pcs = np_col_pcs[:, :2]

    # plot first two principal components and determine number of clusters
    fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
    axes.scatter(np_col_pcs[:, 0][::3], np_col_pcs[:, 1][::3], marker=".", alpha=0.1)
    axes.set_xlabel("First principal component")
    axes.set_ylabel("Second principal component")
    axes.set_title("First and second spike principal components")
    plt.savefig(os.path.join(dir_output, "first and second spike principal components.png"))
    plt.close()

    # visual inspection indicates there are three clusters
    n_clusters = 3

    # perform k-means
    k_means = KMeans(n_clusters=n_clusters)
    np_cluster_indices = k_means.fit(np_col_pcs, center_method="centroid", distance_method="euclidian")
    fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
    axes.scatter(np_col_pcs[:, 0][::3], np_col_pcs[:, 1][::3], c=np_cluster_indices[::3], marker=".", alpha=0.1)
    np_centroid_history = np.array(k_means.np_centroids_history)
    for cluster_idx in range(n_clusters):
        axes.plot(np_centroid_history[:, cluster_idx, 0], np_centroid_history[:, cluster_idx, 1], color="black")
    axes.scatter(k_means.np_centroids[:, 0], k_means.np_centroids[:, 1], color="black")
    axes.set_xlabel("First principal component")
    axes.set_ylabel("Second principal component")
    axes.set_title("k-means")
    plt.savefig(os.path.join(dir_output, "k means.png"))
    plt.close()

    # =========================================================================
    # PART 2.1: K-MEDOIDS CLUSTERING
    # =========================================================================

    # perform k-medoids
    k_means = KMeans(n_clusters=n_clusters)
    np_cluster_indices = k_means.fit(np_col_pcs, center_method="medoid", distance_method="euclidian")
    fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
    axes.scatter(np_col_pcs[:, 0][::3], np_col_pcs[:, 1][::3], c=np_cluster_indices[::3], marker=".", alpha=0.1)
    np_centroid_history = np.array(k_means.np_centroids_history)
    for cluster_idx in range(n_clusters):
        axes.plot(np_centroid_history[:, cluster_idx, 0], np_centroid_history[:, cluster_idx, 1], color="black")
    axes.scatter(k_means.np_centroids[:, 0], k_means.np_centroids[:, 1], color="black")
    axes.set_xlabel("First principal component")
    axes.set_ylabel("Second principal component")
    axes.set_title("k-medoids")
    plt.savefig(os.path.join(dir_output, "k medoids.png"))
    plt.close()

    # =========================================================================
    # PART 2.2: K-MEANS WITH MAHALANOBIS CLUSTERING
    # =========================================================================

    # perform k-medoids
    k_means = KMeans(n_clusters=n_clusters)
    np_cluster_indices = k_means.fit(
        np_col_pcs,
        center_method="centroid", distance_method="mahalanobis",
        convergence_threshold=0.00001)
    fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
    axes.scatter(np_col_pcs[:, 0][::3], np_col_pcs[:, 1][::3], c=np_cluster_indices[::3], marker=".", alpha=0.1)
    np_centroid_history = np.array(k_means.np_centroids_history)
    for cluster_idx in range(n_clusters):
        axes.plot(np_centroid_history[:, cluster_idx, 0], np_centroid_history[:, cluster_idx, 1], color="black")
    axes.scatter(k_means.np_centroids[:, 0], k_means.np_centroids[:, 1], color="black")
    axes.set_xlabel("First principal component")
    axes.set_ylabel("Second principal component")
    axes.set_title("k-means with mahalobis")
    plt.savefig(os.path.join(dir_output, "mahalobis.png"))
    plt.close()


if __name__ == "__main__":
    _filepath_spikes = r"D:\Documents\Academics\BME517\bme_lab_6\data\spikes.mat"
    _dir_output = r"D:\Documents\Academics\BME517\bme_lab_5_6_report"
    main(_filepath_spikes, _dir_output)
