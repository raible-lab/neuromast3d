#!usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import KDTree


class RepresentativeCellFinder(KDTree):
    def __init__(self, data):
        super().__init__(data)

    def query_all_rows(self):
        # Just a test method to get used to how this works
        for row in range(self.data.shape[0]):
            dist, ind = self.query(self.data[row])
            print(dist, ind)

    def find_cells_along_one_column(self, col_idx: int, num_sds: int, k: int):
        # For a given column index, calulcate the col_mean
        # and the col_mean +/- a given number of standard deviations
        # Then, query the KDTree to find k points closest to those values.
        col_values = self.data[:, col_idx]
        col_mean = np.mean(col_values)
        col_sd = np.std(col_values)
        pts_to_query = [col_mean + n*col_sd for n in np.arange(-num_sds, num_sds+1)]

        dists = []
        inds = []
        for pt in pts_to_query:
            array_to_query = np.zeros(self.data.shape[1])
            array_to_query[col_idx] = pt
            dist, ind = self.query(array_to_query, k)
            dists.append(dist)
            inds.append(ind)
        return dists, inds

    def find_cluster_centroids(self, cluster_labels: np.ndarray):
        self.cluster_labels = cluster_labels
        cluster_labels = self.cluster_labels[:, np.newaxis]
        clustered_data = np.concatenate((self.data, cluster_labels), axis=1)
        cluster_means = {}
        for cluster in np.unique(cluster_labels):
            subset = clustered_data[clustered_data[:, -1] == cluster]
            cluster_mean = np.mean(subset[:, :-1], axis=0)
            cluster_means[cluster] = cluster_mean
        self.cluster_means = cluster_means
        return cluster_means

    def find_cells_near_cluster_centroid(self, cluster: int, k: int):
        cluster_mean = self.cluster_means[cluster]
        dists, inds = self.query(cluster_mean, k)
        return dists, inds

    def find_cells_near_all_cluster_centroids(self, k):
        repr_cells = []
        for cluster in np.unique(self.cluster_labels):
            cluster = int(cluster)
            dists, inds = self.find_cells_near_cluster_centroid(self.cluster_labels, cluster, k)
            if k == 1:
                repr_cells.append({'cluster': cluster, 'k': k, 'dists': dists, 'inds': inds})

            else:
                for k_val in range(k):
                    repr_cells.append({'cluster': cluster, 'k': k_val, 'dists': dists[k_val], 'inds': inds[k_val]})

        return repr_cells



def main():
    test_array = np.random.randint(1, 11, (1000, 2000))
    cluster_labels = np.random.randint(0, 4, 1000)
    finder = RepresentativeCellFinder(test_array)
    dists, inds = finder.find_cells_near_cluster_centroid(
        cluster_labels, 0, 3
    )
    print(dists, inds)


if __name__ == '__main__':
    main()
