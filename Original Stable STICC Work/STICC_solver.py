from src.admm_solver import ADMMSolver
from src.STICC_helper import *
from multiprocessing import Pool
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import time
import collections
import os
import errno
import sys
import code
import random
import matplotlib
import itertools
matplotlib.use('Agg')

from sklearn.covariance import GraphicalLasso
from scipy.spatial import distance
from glasso.glasso import miss_glasso


class STICC:
    def __init__(self, spatial_radius=1, number_of_clusters=5, lambda_parameter=11e-2,
                 beta=400, maxIters=1000, threshold=2e-5, write_out_file=False,
                 prefix_string="", num_proc=1, cluster_reassignment=20, biased=False,
                 attr_idx_start=0, attr_idx_end=0, coord_idx_start=0, coord_idx_end=0, spatial_idx_start=0, spatial_idx_end=0):
        """
        Parameters:
            - spatial_radius: size of the subregion
            - number_of_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - threshold: convergence threshold
            - write_out_file: (bool) if true, prefix_string is output file dir
            - prefix_string: output directory if necessary
            - cluster_reassignment: number of points to reassign to a 0 cluster
            - biased: Using the biased or the unbiased covariance
        """
        self.spatial_radius = spatial_radius
        self.number_of_clusters = number_of_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.maxIters = maxIters
        self.threshold = threshold
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.num_proc = num_proc
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.spatial_radius + 1
        self.biased = biased
        self.attr_idx_start = attr_idx_start
        self.attr_idx_end = attr_idx_end
        self.coord_idx_start = coord_idx_start
        self.coord_idx_end = coord_idx_end
        self.spatial_idx_start = spatial_idx_start
        self.spatial_idx_end = spatial_idx_end
        self.spatial_series_index = []
        self.spatial_series_close = []
        self.spatial_series_closest = []
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

    def fit(self, input_file):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(
            input_file)
        print(total_arr.shape)
        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created
        str_NULL = self.prepare_out_directory()

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        # Stack the training data
        complete_D_train = self.stack_training_data(total_arr, spatial_series_col_size, num_train_points,
                                                    training_indices, spatial_series_col_size)

        print("Complete D train shape: ", complete_D_train.shape)

        # Initialization
        # Gaussian Mixture
        #gmm = mixture.GaussianMixture(
        #    n_components=self.number_of_clusters, covariance_type="full")
        #gmm.fit(complete_D_train)
        #clustered_points = gmm.predict(complete_D_train)
        #gmm_clustered_pts = clustered_points + 0
        # K-means
        #kmeans = KMeans(n_clusters=self.number_of_clusters,
        #                n_init=300, random_state=0).fit(complete_D_train)
        print(spatial_series_arr.shape)
        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300, random_state=0).fit(spatial_series_arr)
        clustered_points = kmeans.labels_

        print(collections.Counter(clustered_points))

        # todo, is there a difference between these two?
        #clustered_points_kmeans = kmeans.labels_
        #kmeans_clustered_pts = kmeans.labels_

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': complete_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = complete_D_train[
                                point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            # TEST SETS STUFF
            # LLE + swtiching_penalty
            # Segment length
            # Get the train and test points
            #train_confusion_matrix_EM = compute_confusion_matrix(self.number_of_clusters, clustered_points,
            #                                                     training_indices)
            #train_confusion_matrix_GMM = compute_confusion_matrix(self.number_of_clusters, gmm_clustered_pts,
            #                                                      training_indices)
            #train_confusion_matrix_kmeans = compute_confusion_matrix(self.number_of_clusters, kmeans_clustered_pts,
            #                                                         training_indices)
            # compute the matchings
            #matching_EM, matching_GMM, matching_Kmeans = self.compute_matches(train_confusion_matrix_EM,
            #                                                                  train_confusion_matrix_GMM,
            #                                                                  train_confusion_matrix_kmeans)

            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
            pool.close()
            pool.join()
        #train_confusion_matrix_EM = compute_confusion_matrix(self.number_of_clusters, clustered_points,
        #                                                     training_indices)
        #train_confusion_matrix_GMM = compute_confusion_matrix(self.number_of_clusters, gmm_clustered_pts,
        #                                                      training_indices)
        #train_confusion_matrix_kmeans = compute_confusion_matrix(self.number_of_clusters, clustered_points_kmeans,
        #                                                         training_indices)

        return clustered_points, train_cluster_inverse

    def fit_global_mean(self, input_file):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(
            input_file)
        print(total_arr.shape)
        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created
        str_NULL = self.prepare_out_directory()

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        nidx = np.isnan(spatial_series_arr)
        observed_arr = spatial_series_arr[~nidx.any(axis=1)]
        observed_mean = np.ones_like(spatial_series_arr) * np.mean(observed_arr, axis=0).reshape((1,-1))
        spatial_series_arr[nidx] = 0.0
        spatial_series_arr += observed_mean * nidx

        # Stack the training data
        total_arr = np.hstack((spatial_series_arr, spatial_series_close))
        complete_D_train = self.stack_training_data(total_arr, spatial_series_col_size, num_train_points,
                                                    training_indices, spatial_series_col_size)

        print("Complete D train shape: ", complete_D_train.shape)

        print(spatial_series_arr.shape)
        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300, random_state=0).fit(spatial_series_arr)
        clustered_points = kmeans.labels_

        print(collections.Counter(clustered_points))

        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': complete_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = complete_D_train[
                                point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))


            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
            pool.close()
            pool.join()

        return clustered_points, train_cluster_inverse

    def fit_local_mean(self, input_file, radius=3):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(
            input_file)
        print(total_arr.shape)
        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created
        str_NULL = self.prepare_out_directory()

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        local_means = []
        for nb in total_arr[:,self.spatial_idx_start:self.spatial_idx_start+radius+1]:
            nb_feat = spatial_series_arr[nb.astype(int)]
            nidx = np.isnan(nb_feat)
            observed_arr = nb_feat[~nidx.any(axis=1)]
            if observed_arr.shape[0] == 0:
                observed_mean = np.zeros(nb_feat.shape[1])
            else:
                observed_mean = np.mean(observed_arr, axis=0)
            local_means.append(observed_mean)

        local_means = np.array(local_means)

        nidx = np.isnan(spatial_series_arr)
        spatial_series_arr[nidx] = 0.0
        spatial_series_arr += local_means * nidx

        # Stack the training data
        total_arr = np.hstack((spatial_series_arr, spatial_series_close))
        complete_D_train = self.stack_training_data(total_arr, spatial_series_col_size, num_train_points,
                                                    training_indices, spatial_series_col_size)

        print("Complete D train shape: ", complete_D_train.shape)

        print(spatial_series_arr.shape)
        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(spatial_series_arr)
        clustered_points = kmeans.labels_

        print(collections.Counter(clustered_points))

        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': complete_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = complete_D_train[
                                point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))


            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
            pool.close()
            pool.join()

        return clustered_points, train_cluster_inverse

    def fit_kriging(self, input_file, step=50):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(
            input_file)
        print(total_arr.shape)
        spatial_series_arr = total_arr[:, self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_coords = total_arr[:, self.coord_idx_start:self.coord_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:, self.spatial_idx_start:self.spatial_idx_end+1]

        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created
        str_NULL = self.prepare_out_directory()

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        nidxs = ~np.isnan(spatial_series_arr)
        spatial_series_arr[nidxs==0] = 0.0
        xs, ys = spatial_series_coords[:,0], spatial_series_coords[:,1]
        minx, miny = np.min(xs), np.min(ys)
        maxx, maxy = np.max(xs), np.max(ys)
        #print(minx, maxx, miny, maxy)
        for i in range(self.attr_idx_end+1-self.attr_idx_start):
            nidx = nidxs[:, i]
            xcol, ycol, fcol = xs[nidx == 1], ys[nidx == 1], spatial_series_arr[nidx == 1, i]

            OK = OrdinaryKriging(xcol, ycol, fcol, variogram_model="linear", verbose=False, enable_plotting=False)

            gridx = np.arange(minx-step, maxx+step, step)
            gridy = np.arange(miny-step, maxy+step, step)

            #print(gridx.shape)
            #print(gridy.shape)

            z, ss = OK.execute("grid", gridx, gridy)

            #print(z.shape)

            kxs, kys = ((xs-minx+step) // step).astype(int), ((ys-miny+step) // step).astype(int)
            print(kxs.shape, kys.shape)

            kfit = (z[kys, kxs] + z[kys+1, kxs] + z[kys, kxs+1] + z[kys+1, kxs+1]) / 4
            spatial_series_arr[:,i] += (~nidx) * kfit

        print(spatial_series_arr==0.0)

        # Stack the training data
        total_arr = np.hstack((spatial_series_arr, spatial_series_close))
        complete_D_train = self.stack_training_data(total_arr, spatial_series_col_size, num_train_points,
                                                    training_indices, spatial_series_col_size)

        print("Complete D train shape: ", complete_D_train.shape)

        print(spatial_series_arr.shape)
        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(spatial_series_arr)
        clustered_points = kmeans.labels_

        print(collections.Counter(clustered_points))

        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': complete_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = complete_D_train[
                                point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))


            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
            pool.close()
            pool.join()

        return clustered_points, train_cluster_inverse

    def fit_tree(self, input_file, radius=3):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(
            input_file)
        print(total_arr.shape)
        spatial_series_arr = total_arr[:, self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_coords = total_arr[:, self.coord_idx_start:self.coord_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:, self.spatial_idx_start:self.spatial_idx_end+1]

        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created
        str_NULL = self.prepare_out_directory()

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        nidxs = np.isnan(spatial_series_arr)
        complete_series_arr = spatial_series_arr[(~nidxs).all(axis=1)]
        print("Complete data: ", complete_series_arr.shape)

        max_unknown = (self.attr_idx_end - self.attr_idx_start + 1) / 2
        print(max_unknown)

        local_means = []
        for nb in total_arr[:, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
            nb_feat = spatial_series_arr[nb.astype(int)]
            nidx = np.isnan(nb_feat)
            observed_arr = nb_feat[~nidx.any(axis=1)]
            if observed_arr.shape[0] == 0:
                observed_mean = np.zeros(nb_feat.shape[1])
            else:
                observed_mean = np.mean(observed_arr, axis=0)
            local_means.append(observed_mean)

        local_means = np.array(local_means)

        nidxs[np.sum(nidxs, axis=1) < max_unknown] = 0
        spatial_series_arr[nidxs] = 0.0
        spatial_series_arr += local_means * nidxs

        nidxs = np.isnan(spatial_series_arr)

        D = spatial_series_arr.shape[1]
        lst = list(itertools.product([0, 1], repeat=D))

        for t in lst:
            tidx = np.array(t).astype(bool)
            d = np.sum(t)
            if d < max_unknown and d > 0:
                X, y = complete_series_arr[:,~tidx].reshape((-1,D-d)), complete_series_arr[:,tidx].reshape((-1,d))

                regr = DecisionTreeRegressor(max_depth=5)
                regr.fit(X, y)

                aidx = (np.sum(nidxs[:,tidx], axis=1) == d)
                bidx = (np.sum(nidxs[:,~tidx], axis=1) == 0)
                midx = aidx & bidx

                if (~midx).all():
                    continue

                X_pred = spatial_series_arr[midx]
                X_pred = X_pred[:,~tidx]
                y_pred = regr.predict(X_pred)

                tmp = spatial_series_arr[midx]
                tmp[:,tidx] = y_pred.reshape((-1,d))

                spatial_series_arr[midx] = tmp

        # Stack the training data
        total_arr = np.hstack((spatial_series_arr, spatial_series_close))
        complete_D_train = self.stack_training_data(total_arr, spatial_series_col_size, num_train_points,
                                                    training_indices, spatial_series_col_size)

        print("Complete D train shape: ", complete_D_train.shape)

        print(spatial_series_arr.shape)
        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(spatial_series_arr)
        clustered_points = kmeans.labels_

        print(collections.Counter(clustered_points))

        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': complete_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = complete_D_train[
                                point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))


            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
            pool.close()
            pool.join()

        return clustered_points, train_cluster_inverse

    def fit_missglasso(self, input_file, radius=3):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(input_file)

        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        # Stack the training data
        #imputed_arr = self.initial_missglasso_imputation(total_arr)
        missing_D_train = self.stack_training_data(total_arr, spatial_series_col_size, num_train_points,
                                                    training_indices, spatial_series_col_size)

        # print(missing_D_train.shape)

        # Initialization
        # Gaussian Mixture
        #gmm = mixture.GaussianMixture(n_components=self.number_of_clusters, covariance_type="full")
        #gmm.fit(complete_D_train)
        #clustered_points = gmm.predict(complete_D_train)
        #gmm_clustered_pts = clustered_points + 0
        # K-means

        ### Two methods: 1) use MissGLasso imputed data to compute initial KNN; 2) use only valid entries to compute KNN

        #complete_arr = self.initial_missglasso_imputation(total_arr)
        #complete_D_train = self.stack_training_data(complete_arr, spatial_series_col_size, num_train_points, training_indices, spatial_series_col_size)
        #complete_D_train = self.initial_missglasso_imputation(missing_D_train)
        local_means = []
        for nb in total_arr[:, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
            nb_feat = spatial_series_arr[nb.astype(int)]
            nidx = np.isnan(nb_feat)
            observed_arr = nb_feat[~nidx.any(axis=1)]
            if observed_arr.shape[0] == 0:
                observed_mean = np.zeros(nb_feat.shape[1])
            else:
                observed_mean = np.mean(observed_arr, axis=0)
            local_means.append(observed_mean)

        local_means = np.array(local_means)

        nidx = np.isnan(spatial_series_arr)
        nidx[np.sum(nidx, axis=1)<(self.attr_idx_end - self.attr_idx_start + 1)/2] = 0
        spatial_series_arr[nidx] = 0.0
        spatial_series_arr += local_means * nidx

        complete_arr = self.missglasso_imputation(spatial_series_arr)

        print(complete_arr.shape)

        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(complete_arr)
        clustered_points = kmeans.labels_
        print(collections.Counter(clustered_points))



        # todo, is there a difference between these two?
        #clustered_points_kmeans = kmeans.labels_
        #kmeans_clustered_pts = kmeans.labels_

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            #### M step: update the MRF covariance
            # imputed_D_train = self.solve_missglasso(cluster_mean_info, cluster_mean_stacked_info, missing_D_train, len_train_clusters, spatial_series_col_size, train_clusters_arr, train_cluster_inverse, computed_covariance)

            imputed_series_arr = self.solve_missglasso(spatial_series_arr, len_train_clusters, spatial_series_col_size, train_clusters_arr)

            print("Imputed Series ARR", imputed_series_arr.shape)

            imputed_total_arr = np.hstack((imputed_series_arr, spatial_series_close))
            print("Imputed Total ARR", imputed_total_arr.shape)
            imputed_D_train = self.stack_training_data(imputed_total_arr, spatial_series_col_size, num_train_points, training_indices, spatial_series_col_size)

            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, imputed_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': imputed_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}

            #clustered_points = self.predict_clusters(complete_D_train)
            #print(collections.Counter(clustered_points))

            clustered_points = self.predict_clusters()


            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = imputed_D_train[
                               point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                               = imputed_D_train[point_to_move, :][
                               (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
           pool.close()
           pool.join()

        return clustered_points, train_cluster_inverse

    def fit_missglasso_static(self, input_file, interp="missglasso", radius=3, thres=5.68, rate=0.5, randomize="None"):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(input_file)

        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_coords = total_arr[:, self.coord_idx_start:self.coord_idx_end + 1]
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        # Estimate defective data
        D = self.attr_idx_end - self.attr_idx_start + 1
        ext_id = self.estimate_extremity(spatial_series_arr, spatial_series_close, spatial_series_arr, mala_thres=thres)
        print("Extremity number: ", len(ext_id))

        if randomize == "initial":
            ext_id = np.random.choice(spatial_series_arr.shape[0], len(ext_id))

        if interp == "missglasso":
            spatial_series_arr[ext_id] = np.nan
            local_means = []
            for nb in total_arr[ext_id, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
                nb_feat = spatial_series_arr[nb.astype(int)]
                nidx = np.isnan(nb_feat)
                observed_arr = nb_feat[~nidx.any(axis=1)]
                if observed_arr.shape[0] == 0:
                    observed_mean = np.zeros(nb_feat.shape[1])
                else:
                    observed_mean = np.mean(observed_arr, axis=0)
                local_means.append(observed_mean)

            local_means = np.array(local_means)

            mask = (np.random.random(local_means.shape) < rate).astype(int)
            local_means[mask==1] = np.nan

            spatial_series_arr[ext_id] = local_means
            spatial_series_arr = self.missglasso_imputation(spatial_series_arr)
        elif interp == "globalmean":
            global_mean = np.mean(spatial_series_arr[list(set(range(spatial_series_arr.shape[0])) - set(ext_id))], axis=0).reshape((1,-1))
            print(global_mean)
            spatial_series_arr[ext_id] = np.ones((len(ext_id), D)) * global_mean
        elif interp == "localmean":
            spatial_series_arr[ext_id] = np.nan
            local_means = []
            for nb in total_arr[:, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
                nb_feat = spatial_series_arr[nb.astype(int)]
                nidx = np.isnan(nb_feat)
                observed_arr = nb_feat[~nidx.any(axis=1)]
                if observed_arr.shape[0] == 0:
                    observed_mean = np.zeros(nb_feat.shape[1])
                else:
                    observed_mean = np.mean(observed_arr, axis=0)
                local_means.append(observed_mean)

            local_means = np.array(local_means)

            nidx = np.isnan(spatial_series_arr)
            nidx[np.sum(nidx, axis=1) < (self.attr_idx_end - self.attr_idx_start + 1) / 2] = 0
            spatial_series_arr[nidx] = 0.0
            spatial_series_arr += local_means * nidx
        elif interp == "kriging":
            spatial_series_arr[ext_id] = np.nan
            nidxs = ~np.isnan(spatial_series_arr)
            spatial_series_arr[nidxs == 0] = 0.0
            xs, ys = spatial_series_coords[:, 0], spatial_series_coords[:, 1]
            minx, miny = np.min(xs), np.min(ys)
            maxx, maxy = np.max(xs), np.max(ys)
            # print(minx, maxx, miny, maxy)
            for i in range(self.attr_idx_end + 1 - self.attr_idx_start):
                nidx = nidxs[:, i]
                xcol, ycol, fcol = xs[nidx == 1], ys[nidx == 1], spatial_series_arr[nidx == 1, i]

                OK = OrdinaryKriging(xcol, ycol, fcol, variogram_model="linear", verbose=False, enable_plotting=False)

                gridx = np.arange(minx - 50, maxx + 50, 50)
                gridy = np.arange(miny - 50, maxy + 50, 50)

                z, ss = OK.execute("grid", gridx, gridy)

                kxs, kys = ((xs - minx + 50) // 50).astype(int), ((ys - miny + 50) // 50).astype(int)
                print(kxs.shape, kys.shape)

                kfit = (z[kys, kxs] + z[kys + 1, kxs] + z[kys, kxs + 1] + z[kys + 1, kxs + 1]) / 4
                spatial_series_arr[:, i] += (~nidx) * kfit
        elif interp == "tree":
            tmp = spatial_series_arr[ext_id]
            mask = (np.random.random(tmp.shape) < rate).astype(int)
            tmp[mask==1] = np.nan
            spatial_series_arr[ext_id] = tmp

            nidxs = np.isnan(spatial_series_arr)
            complete_series_arr = spatial_series_arr[(~nidxs).all(axis=1)]
            print("Complete data: ", complete_series_arr.shape)

            max_unknown = (self.attr_idx_end - self.attr_idx_start + 1) / 2
            print(max_unknown)

            local_means = []
            for nb in total_arr[:, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
                nb_feat = spatial_series_arr[nb.astype(int)]
                nidx = np.isnan(nb_feat)
                observed_arr = nb_feat[~nidx.any(axis=1)]
                if observed_arr.shape[0] == 0:
                    observed_mean = np.zeros(nb_feat.shape[1])
                else:
                    observed_mean = np.mean(observed_arr, axis=0)
                local_means.append(observed_mean)

            local_means = np.array(local_means)

            nidxs[np.sum(nidxs, axis=1) < max_unknown] = 0
            spatial_series_arr[nidxs] = 0.0
            spatial_series_arr += local_means * nidxs

            nidxs = np.isnan(spatial_series_arr)

            D = spatial_series_arr.shape[1]
            lst = list(itertools.product([0, 1], repeat=D))

            for t in lst:
                tidx = np.array(t).astype(bool)
                d = np.sum(t)
                if d < max_unknown and d > 0:
                    X, y = complete_series_arr[:, ~tidx].reshape((-1, D - d)), complete_series_arr[:, tidx].reshape(
                        (-1, d))

                    regr = DecisionTreeRegressor(max_depth=5)
                    regr.fit(X, y)

                    aidx = (np.sum(nidxs[:, tidx], axis=1) == d)
                    bidx = (np.sum(nidxs[:, ~tidx], axis=1) == 0)
                    midx = aidx & bidx

                    if (~midx).all():
                        continue

                    X_pred = spatial_series_arr[midx]
                    X_pred = X_pred[:, ~tidx]
                    y_pred = regr.predict(X_pred)

                    tmp = spatial_series_arr[midx]
                    tmp[:, tidx] = y_pred.reshape((-1, d))

                    spatial_series_arr[midx] = tmp
        else:
            assert False, "No supported interpolation method!"



        imputed_total_arr = np.hstack((spatial_series_arr, spatial_series_close))
        imputed_D_train = self.stack_training_data(imputed_total_arr, spatial_series_col_size, num_train_points,
                                                   training_indices, spatial_series_col_size)

        #complete_arr = self.missglasso_imputation(spatial_series_arr)

        #kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(complete_arr)

        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300, random_state=0).fit(spatial_series_arr)
        clustered_points = kmeans.labels_
        print(collections.Counter(clustered_points))



        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            spatial_series_arr[ext_id] = np.nan
            local_means = []
            for nb in total_arr[ext_id, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
                nb_feat = spatial_series_arr[nb.astype(int)]
                nidx = np.isnan(nb_feat)
                observed_arr = nb_feat[~nidx.any(axis=1)]
                if observed_arr.shape[0] == 0:
                    observed_mean = np.zeros(nb_feat.shape[1])
                else:
                    observed_mean = np.mean(observed_arr, axis=0)
                local_means.append(observed_mean)

            local_means = np.array(local_means)

            mask = (np.random.random(local_means.shape) < rate).astype(int)

            local_means[mask == 1] = np.nan

            spatial_series_arr[ext_id] = local_means
            spatial_series_arr = self.missglasso_imputation(spatial_series_arr)

            imputed_total_arr = np.hstack((spatial_series_arr, spatial_series_close))
            imputed_D_train = self.stack_training_data(imputed_total_arr, spatial_series_col_size, num_train_points,
                                                       training_indices, spatial_series_col_size)

            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, imputed_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': imputed_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}


            #clustered_points = self.predict_clusters(complete_D_train)
            #print(collections.Counter(clustered_points))

            clustered_points = self.predict_clusters()


            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = imputed_D_train[
                               point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                               = imputed_D_train[point_to_move, :][
                               (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
           pool.close()
           pool.join()

        return clustered_points, train_cluster_inverse

    def fit_missglasso_static_missing(self, input_file, interp="globalmean", radius=3):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(input_file)

        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_coords = total_arr[:, self.coord_idx_start:self.coord_idx_end + 1]
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        # Estimate defective data
        D = self.attr_idx_end - self.attr_idx_start + 1
        mask = (np.isnan(spatial_series_arr)).astype(int)
        part_idx = np.where(np.sum(mask, axis=1)>0)[0]
        comp_idx = np.where(np.sum(mask, axis=1)==0)[0]

        if interp == "globalmean":
            global_mean = np.mean(spatial_series_arr[comp_idx], axis=0).reshape((1,-1))
            print(global_mean)
            spatial_series_arr[part_idx] = global_mean
        elif interp == "localmean":
            local_means = []
            for nb in total_arr[part_idx, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
                nb_feat = spatial_series_arr[nb.astype(int)]
                nidx = np.isnan(nb_feat)
                observed_arr = nb_feat[~nidx.any(axis=1)]
                if observed_arr.shape[0] == 0:
                    observed_mean = np.zeros(nb_feat.shape[1])
                else:
                    observed_mean = np.mean(observed_arr, axis=0)
                local_means.append(observed_mean)

            local_means = np.array(local_means)

            spatial_series_arr[part_idx] = local_means
        elif interp == "kriging":
            nidxs = ~np.isnan(spatial_series_arr)
            spatial_series_arr[nidxs == 0] = 0.0
            xs, ys = spatial_series_coords[:, 0], spatial_series_coords[:, 1]
            minx, miny = np.min(xs), np.min(ys)
            maxx, maxy = np.max(xs), np.max(ys)
            # print(minx, maxx, miny, maxy)
            for i in range(self.attr_idx_end + 1 - self.attr_idx_start):
                nidx = nidxs[:, i]
                xcol, ycol, fcol = xs[nidx == 1], ys[nidx == 1], spatial_series_arr[nidx == 1, i]

                OK = OrdinaryKriging(xcol, ycol, fcol, variogram_model="linear", verbose=False, enable_plotting=False)

                gridx = np.arange(minx - 50, maxx + 50, 50)
                gridy = np.arange(miny - 50, maxy + 50, 50)

                z, ss = OK.execute("grid", gridx, gridy)

                kxs, kys = ((xs - minx + 50) // 50).astype(int), ((ys - miny + 50) // 50).astype(int)
                print(kxs.shape, kys.shape)

                kfit = (z[kys, kxs] + z[kys + 1, kxs] + z[kys, kxs + 1] + z[kys + 1, kxs + 1]) / 4
                spatial_series_arr[:, i] += (~nidx) * kfit
        elif interp == "tree":
            nidxs = np.isnan(spatial_series_arr)
            complete_series_arr = spatial_series_arr[(~nidxs).all(axis=1)]
            print("Complete data: ", complete_series_arr.shape)

            max_unknown = (self.attr_idx_end - self.attr_idx_start + 1) / 2
            print(max_unknown)

            local_means = []
            for nb in total_arr[:, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
                nb_feat = spatial_series_arr[nb.astype(int)]
                nidx = np.isnan(nb_feat)
                observed_arr = nb_feat[~nidx.any(axis=1)]
                if observed_arr.shape[0] == 0:
                    observed_mean = np.zeros(nb_feat.shape[1])
                else:
                    observed_mean = np.mean(observed_arr, axis=0)
                local_means.append(observed_mean)

            local_means = np.array(local_means)

            nidxs[np.sum(nidxs, axis=1) < max_unknown] = 0
            spatial_series_arr[nidxs] = 0.0
            spatial_series_arr += local_means * nidxs

            nidxs = np.isnan(spatial_series_arr)

            D = spatial_series_arr.shape[1]
            lst = list(itertools.product([0, 1], repeat=D))

            for t in lst:
                tidx = np.array(t).astype(bool)
                d = np.sum(t)
                if d < max_unknown and d > 0:
                    X, y = complete_series_arr[:, ~tidx].reshape((-1, D - d)), complete_series_arr[:, tidx].reshape(
                        (-1, d))

                    regr = DecisionTreeRegressor(max_depth=5)
                    regr.fit(X, y)

                    aidx = (np.sum(nidxs[:, tidx], axis=1) == d)
                    bidx = (np.sum(nidxs[:, ~tidx], axis=1) == 0)
                    midx = aidx & bidx

                    if (~midx).all():
                        continue

                    X_pred = spatial_series_arr[midx]
                    X_pred = X_pred[:, ~tidx]
                    y_pred = regr.predict(X_pred)

                    tmp = spatial_series_arr[midx]
                    tmp[:, tidx] = y_pred.reshape((-1, d))

                    spatial_series_arr[midx] = tmp
        else:
            assert False, "No supported interpolation method!"



        imputed_total_arr = np.hstack((spatial_series_arr, spatial_series_close))
        imputed_D_train = self.stack_training_data(imputed_total_arr, spatial_series_col_size, num_train_points,
                                                   training_indices, spatial_series_col_size)

        #complete_arr = self.missglasso_imputation(spatial_series_arr)

        #print(complete_arr.shape)

        #kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(complete_arr)
        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(spatial_series_arr)
        clustered_points = kmeans.labels_
        print(collections.Counter(clustered_points))



        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            spatial_series_arr = self.missglasso_imputation(spatial_series_arr)

            imputed_total_arr = np.hstack((spatial_series_arr, spatial_series_close))
            imputed_D_train = self.stack_training_data(imputed_total_arr, spatial_series_col_size, num_train_points,
                                                       training_indices, spatial_series_col_size)

            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, imputed_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': imputed_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}

            #clustered_points = self.predict_clusters(complete_D_train)
            #print(collections.Counter(clustered_points))

            clustered_points = self.predict_clusters()


            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = imputed_D_train[
                               point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                               = imputed_D_train[point_to_move, :][
                               (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
           pool.close()
           pool.join()

        return clustered_points, train_cluster_inverse

    def fit_missglasso_dynamic(self, input_file, radius=3, initial_thres=5.68, thres=5.68, rate=0.50, randomize="None"):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(input_file)

        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        # Estimate defective data
        ext_id = self.estimate_extremity(spatial_series_arr, spatial_series_close, spatial_series_arr, mala_thres=initial_thres)
        print("Extremity number: ", len(ext_id))

        if randomize == "initial" or randomize == "complete":
            ext_id = np.random.choice(spatial_series_arr.shape[0], len(ext_id))

        spatial_series_arr[ext_id] = np.nan
        local_means = []
        for nb in total_arr[ext_id, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
            nb_feat = spatial_series_arr[nb.astype(int)]
            nidx = np.isnan(nb_feat)
            observed_arr = nb_feat[~nidx.any(axis=1)]
            if observed_arr.shape[0] == 0:
                observed_mean = np.zeros(nb_feat.shape[1])
            else:
                observed_mean = np.mean(observed_arr, axis=0)
            local_means.append(observed_mean)

        local_means = np.array(local_means)

        spatial_series_arr[ext_id] = copy.deepcopy(local_means)
        # spatial_series_arr = self.missglasso_imputation(spatial_series_arr)

        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(spatial_series_arr)
        clustered_points = kmeans.labels_
        print(collections.Counter(clustered_points))



        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            #### M step: update the MRF covariance
            # mask = (np.random.random(local_means.shape) < rate).astype(int)
            # tmp = copy.deepcopy(local_means)
            # tmp[mask==1] = np.nan
            #
            # spatial_series_arr[ext_id] = tmp
            # print(len(np.where(np.isnan(spatial_series_arr))[0]))
            # spatial_series_arr = self.missglasso_imputation(spatial_series_arr)

            for i in range(self.number_of_clusters):
                ctmp = spatial_series_arr[clustered_points==i]
                #cnbs = spatial_series_close[clustered_points==i]
                ext_id = self.estimate_extremity(ctmp, spatial_series_close, spatial_series_arr, mala_thres=thres)
                print("Extremity number for cluster {}: {} out of {}".format(i, len(ext_id), ctmp.shape[0]))

                if randomize == "iterative" or randomize == "complete":
                    ext_id = np.random.choice(ctmp.shape[0], len(ext_id))

                cext = copy.deepcopy(ctmp[ext_id])

                rmask = (np.random.random(cext.shape) < rate).astype(int)
                cext[rmask==1] = np.nan
                ctmp[ext_id] = cext

                ctmp = self.missglasso_imputation(ctmp)
                spatial_series_arr[clustered_points == i] = ctmp

            imputed_total_arr = np.hstack((spatial_series_arr, spatial_series_close))
            imputed_D_train = self.stack_training_data(imputed_total_arr, spatial_series_col_size, num_train_points,
                                                       training_indices, spatial_series_col_size)

            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, imputed_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': imputed_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}

            #clustered_points = self.predict_clusters(complete_D_train)
            #print(collections.Counter(clustered_points))

            clustered_points = self.predict_clusters()


            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = imputed_D_train[
                               point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                               = imputed_D_train[point_to_move, :][
                               (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
           pool.close()
           pool.join()

        return clustered_points, train_cluster_inverse

    def fit_missglasso_mixed(self, input_file, radius=3, thres=5.68, rate=0.2):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(input_file)

        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_coords = total_arr[:, self.coord_idx_start:self.coord_idx_end + 1]
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        # Estimate defective data
        D = self.attr_idx_end - self.attr_idx_start + 1
        mask = (np.isnan(spatial_series_arr)).astype(int)
        part_idx = np.where(np.sum(mask, axis=1)>0)[0]
        comp_idx = np.where(np.sum(mask, axis=1)==0)[0]

        local_means = []
        for nb in total_arr[part_idx, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
            nb_feat = spatial_series_arr[nb.astype(int)]
            nidx = np.isnan(nb_feat)
            observed_arr = nb_feat[~nidx.any(axis=1)]
            if observed_arr.shape[0] == 0:
                observed_mean = np.zeros(nb_feat.shape[1])
            else:
                observed_mean = np.mean(observed_arr, axis=0)
            local_means.append(observed_mean)

        local_means = np.array(local_means)

        spatial_series_arr[part_idx] = local_means

        ext_id = self.estimate_extremity(spatial_series_arr, spatial_series_close, spatial_series_arr, mala_thres=thres)
        print("Extremity number: ", len(ext_id))

        spatial_series_arr[ext_id] = np.nan
        local_means = []
        for nb in total_arr[ext_id, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
            nb_feat = spatial_series_arr[nb.astype(int)]
            nidx = np.isnan(nb_feat)
            observed_arr = nb_feat[~nidx.any(axis=1)]
            if observed_arr.shape[0] == 0:
                observed_mean = np.zeros(nb_feat.shape[1])
            else:
                observed_mean = np.mean(observed_arr, axis=0)
            local_means.append(observed_mean)

        local_means = np.array(local_means)

        spatial_series_arr[ext_id] = copy.deepcopy(local_means)

        imputed_total_arr = np.hstack((spatial_series_arr, spatial_series_close))
        imputed_D_train = self.stack_training_data(imputed_total_arr, spatial_series_col_size, num_train_points,
                                                   training_indices, spatial_series_col_size)

        #complete_arr = self.missglasso_imputation(spatial_series_arr)

        #print(complete_arr.shape)

        #kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(complete_arr)
        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(spatial_series_arr)
        clustered_points = kmeans.labels_
        print(collections.Counter(clustered_points))



        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            spatial_series_arr[mask==1] = np.nan
            spatial_series_arr = self.missglasso_imputation(spatial_series_arr)

            for i in range(self.number_of_clusters):
                ctmp = spatial_series_arr[clustered_points==i]
                #cnbs = spatial_series_close[clustered_points==i]
                ext_id = self.estimate_extremity(ctmp, spatial_series_close, spatial_series_arr, mala_thres=thres)
                print("Extremity number for cluster {}: {} out of {}".format(i, len(ext_id), ctmp.shape[0]))

                cext = copy.deepcopy(ctmp[ext_id])

                rmask = (np.random.random(cext.shape) < rate).astype(int)
                cext[rmask==1] = np.nan
                ctmp[ext_id] = cext

                ctmp = self.missglasso_imputation(ctmp)
                spatial_series_arr[clustered_points == i] = ctmp

            imputed_total_arr = np.hstack((spatial_series_arr, spatial_series_close))
            imputed_D_train = self.stack_training_data(imputed_total_arr, spatial_series_col_size, num_train_points,
                                                       training_indices, spatial_series_col_size)

            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, imputed_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': imputed_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}

            #clustered_points = self.predict_clusters(complete_D_train)
            #print(collections.Counter(clustered_points))

            clustered_points = self.predict_clusters()


            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = imputed_D_train[
                               point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                               = imputed_D_train[point_to_move, :][
                               (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
           pool.close()
           pool.join()

        return clustered_points, train_cluster_inverse

    def fit_missglasso_random(self, input_file, rate=0.0, init_random=False, interp="missglasso", radius=3):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(input_file)

        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        print(init_random)

        # Initial random mask and imputation
        if init_random:
            print("Initial random fitting")
            mask = (np.random.random(spatial_series_arr.shape)<rate).astype(int)
            random_arr = copy.deepcopy(spatial_series_arr)
            random_arr[mask==1] = np.nan
            if interp == "missglasso":
                local_means = []
                for nb in total_arr[:, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
                    nb_feat = random_arr[nb.astype(int)]
                    nidx = np.isnan(nb_feat)
                    observed_arr = nb_feat[~nidx.any(axis=1)]
                    if observed_arr.shape[0] == 0:
                        observed_mean = np.zeros(nb_feat.shape[1])
                    else:
                        observed_mean = np.mean(observed_arr, axis=0)
                    local_means.append(observed_mean)

                local_means = np.array(local_means)

                nidx = np.isnan(random_arr)
                nidx[np.sum(nidx, axis=1) < (self.attr_idx_end - self.attr_idx_start + 1) / 2] = 0
                random_arr[nidx] = 0.0
                random_arr += local_means * nidx
                imputed_series_arr = self.missglasso_imputation(random_arr)
            elif interp == "global_mean":
                imputed_series_arr = self.global_mean_imputation(spatial_series_arr)
            elif interp == "local_mean":
                imputed_series_arr = self.local_mean_imputation(spatial_series_arr, total_arr, radius)
            kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(imputed_series_arr)
        else:
            print("No initial random fitting")
            kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300).fit(spatial_series_arr)
        clustered_points = kmeans.labels_
        print(collections.Counter(clustered_points))



        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            mask = (np.random.random(spatial_series_arr.shape) < rate).astype(int)
            #imputed_series_arr[mask==1] = np.nan
            random_arr = copy.deepcopy(spatial_series_arr)
            random_arr[mask == 1] = np.nan

            if interp == "missglasso":
                imputed_series_arr = self.solve_missglasso(random_arr, len_train_clusters, spatial_series_col_size, train_clusters_arr)
            elif interp == "global_mean":
                imputed_series_arr = self.global_mean_imputation(random_arr)
            elif interp == "local_mean":
                imputed_series_arr = self.local_mean_imputation(random_arr, total_arr, radius)


            print("Imputed Series ARR", imputed_series_arr.shape)

            imputed_total_arr = np.hstack((imputed_series_arr, spatial_series_close))
            print("Imputed Total ARR", imputed_total_arr.shape)
            imputed_D_train = self.stack_training_data(imputed_total_arr, spatial_series_col_size, num_train_points, training_indices, spatial_series_col_size)

            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, imputed_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': imputed_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}


            clustered_points = self.predict_clusters()


            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = imputed_D_train[
                               point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                               = imputed_D_train[point_to_move, :][
                               (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
           pool.close()
           pool.join()

        return clustered_points, train_cluster_inverse

    def fit_missglasso_only_once(self, input_file):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format

        total_arr, total_rows_size, total_cols_size = self.load_data(
            input_file)
        spatial_series_arr = total_arr[:,
                                       self.attr_idx_start:self.attr_idx_end+1]
        spatial_series_rows_size = total_rows_size
        spatial_series_col_size = self.attr_idx_end - self.attr_idx_start + 1
        spatial_series_index = total_arr[:, 0]
        spatial_series_close = total_arr[:,
                                         self.spatial_idx_start:self.spatial_idx_end+1]
        print(spatial_series_col_size, spatial_series_arr.shape,
              spatial_series_close.shape)
        self.spatial_series_closest = spatial_series_close[:, 0]
        self.spatial_series_index = spatial_series_index
        self.spatial_series_close = spatial_series_close

        ############
        # The basic folder to be created
        str_NULL = self.prepare_out_directory()

        # Train test split
        training_indices = spatial_series_index
        num_train_points = len(training_indices)

        # Stack the training data
        missing_D_train = self.stack_training_data(total_arr, spatial_series_col_size, num_train_points,
                                                   training_indices, spatial_series_col_size)

        print(missing_D_train.shape)

        complete_D_train = self.initial_missglasso_imputation(missing_D_train)

        print(complete_D_train.shape)

        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=300, random_state=0).fit(complete_D_train)
        clustered_points = kmeans.labels_
        print(collections.Counter(clustered_points))

        # todo, is there a difference between these two?

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, spatial_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': complete_D_train,
                                  'spatial_series_col_size': spatial_series_col_size}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        # a cluster that is not len 0
                        cluster_selected = valid_clusters[counter]
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num,
                              "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = complete_D_train[
                                point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                (self.spatial_radius - 1) * spatial_series_col_size:self.spatial_radius * spatial_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
            pool.close()
            pool.join()

        return clustered_points, train_cluster_inverse

    def compute_matches(self, train_confusion_matrix_EM, train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        matching_Kmeans = find_matching(train_confusion_matrix_kmeans)
        matching_GMM = find_matching(train_confusion_matrix_GMM)
        matching_EM = find_matching(train_confusion_matrix_EM)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.number_of_clusters):
            matched_cluster_e_m = matching_EM[cluster]
            matched_cluster_g_m_m = matching_GMM[cluster]
            matched_cluster_k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster,
                                                     matched_cluster_e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster,
                                                        matched_cluster_g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster,
                                                             matched_cluster_k_means]
        return matching_EM, matching_GMM, matching_Kmeans

    def smoothen_clusters(self, cluster_mean_info, computed_covariance,
                          cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        for cluster in range(self.number_of_clusters):
            cov_matrix = computed_covariance[self.number_of_clusters, cluster][0:(2 * (self.num_blocks - 1)-1) * n,
                                                                               0:(2 * (self.num_blocks - 1)-1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix)
                                 )  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        print("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros(
            [clustered_points_len, self.number_of_clusters])
        for point in range(clustered_points_len):
            if point + self.spatial_radius - 1 < complete_D_train.shape[0]:
                for cluster in range(self.number_of_clusters):
                    cluster_mean = cluster_mean_info[self.number_of_clusters, cluster]
                    cluster_mean_stacked = cluster_mean_stacked_info[self.number_of_clusters, cluster]
                    x = complete_D_train[point, :] - \
                        cluster_mean_stacked[0:(
                            2 * (self.num_blocks - 1)-1) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, (self.spatial_radius) * n]),
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.spatial_radius), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle

        return LLE_all_points_clusters

    def optimize_clusters(self, computed_covariance, len_train_clusters, log_det_values, optRes, train_cluster_inverse):
        for cluster in range(self.number_of_clusters):
            if optRes[cluster] == None:
                continue
            val = optRes[cluster].get()
            print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")
            # THIS IS THE SOLUTION
            S_est = upperToFull(val, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[self.number_of_clusters,
                           cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[self.number_of_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2
        for cluster in range(self.number_of_clusters):
            print("length of the cluster ", cluster,
                  "------>", len_train_clusters[cluster])

    def train_clusters(self, cluster_mean_info, cluster_mean_stacked_info, complete_D_train, empirical_covariances,
                       len_train_clusters, n, pool, train_clusters_arr):
        optRes = [None for i in range(self.number_of_clusters)]
        for cluster in range(self.number_of_clusters):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                size_blocks = n
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, (self.spatial_radius) * n])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                cluster_mean_info[self.number_of_clusters, cluster] = np.mean(D_train, axis=0)[
                    (
                        self.spatial_radius - 1) * n:self.spatial_radius * n].reshape(
                    [1, n])
                cluster_mean_stacked_info[self.number_of_clusters, cluster] = np.mean(
                    D_train, axis=0)

                # Fit a model - OPTIMIZATION
                probSize = (self.spatial_radius) * size_blocks
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                S = np.cov(np.transpose(D_train), bias=self.biased)
                empirical_covariances[cluster] = S

                rho = 1
                solver = ADMMSolver(
                    lamb, (self.spatial_radius), size_blocks, 1, S)
                # apply to process pool
                optRes[cluster] = pool.apply_async(
                    solver, (1000, 1e-6, 1e-6, False,))

        return optRes

    def global_mean_imputation(self, spatial_series_arr):
        nidx = np.isnan(spatial_series_arr)
        observed_arr = spatial_series_arr[~nidx.any(axis=1)]
        observed_mean = np.ones_like(spatial_series_arr) * np.mean(observed_arr, axis=0).reshape((1, -1))
        spatial_series_arr[nidx] = 0.0
        spatial_series_arr += observed_mean * nidx

        return spatial_series_arr

    def local_mean_imputation(self, spatial_series_arr, total_arr, radius):
        local_means = []
        for nb in total_arr[:, self.spatial_idx_start:self.spatial_idx_start + radius + 1]:
            nb_feat = spatial_series_arr[nb.astype(int)]
            nidx = np.isnan(nb_feat)
            observed_arr = nb_feat[~nidx.any(axis=1)]
            if observed_arr.shape[0] == 0:
                observed_mean = np.zeros(nb_feat.shape[1])
            else:
                observed_mean = np.mean(observed_arr, axis=0)
            local_means.append(observed_mean)

        local_means = np.array(local_means)

        nidx = np.isnan(spatial_series_arr)
        spatial_series_arr[nidx] = 0.0
        spatial_series_arr += local_means * nidx

        return spatial_series_arr

    def missglasso_imputation(self, missing_D_train):
        mask = np.ones_like(missing_D_train)
        mask[np.isnan(missing_D_train)] = 0
        covariance, inv_covariance, imputed_data = miss_glasso(missing_D_train, mask, em_iter=20, glasso_lambda=self.lambda_parameter, glasso_iter=2000)
        return imputed_data

    def initial_missglasso_imputation(self, missing_D_train):
        mask = np.ones_like(missing_D_train)
        mask[np.isnan(missing_D_train)] = 0
        covariance, inv_covariance, imputed_data = miss_glasso(missing_D_train, mask, em_iter=20, glasso_lambda=self.lambda_parameter, glasso_iter=2000)
        return imputed_data

    def estimate_extremity(self, features, neighbors, full_features, mala_thres=9.236):
        ext_id = []
        for i in range(features.shape[0]):
            nid = neighbors[i].astype(int)
            s = features[i]
            X = full_features[nid]
            #Y = X + np.random.normal(loc=0.0, scale=0.1, size=X.shape)
            esti = GraphicalLasso().fit(X)
            loc, cov = esti.location_, esti.covariance_

            inv_cov = np.linalg.inv(cov)
            mala_dis = distance.mahalanobis(u=loc, v=s, VI=inv_cov)

            if mala_dis > np.sqrt(mala_thres):
                ext_id.append(i)

        return ext_id

    def solve_missglasso(self, missing_data, len_train_clusters, n, train_clusters_arr):
        imputed_D_train = copy.deepcopy(missing_data)
        for cluster in range(self.number_of_clusters):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, n])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = missing_data[point, :]

                # Fit a model - OPTIMIZATION
                mask = np.ones_like(D_train)
                mask[np.isnan(D_train)] = 0
                # print("D train shape: ", D_train.shape)
                # print("mask shape", mask.shape)
                covariance, inv_covariance, imputed_data = miss_glasso(D_train, mask, em_iter=20, glasso_lambda=self.lambda_parameter, glasso_iter=2000)
                # print("Covariance shape: ", covariance.shape)

                for i in range(cluster_length):
                    point = indices[i]
                    imputed_D_train[point, :] = imputed_data[i, :]

        return imputed_D_train

    def stack_training_data(self, Data, n, num_train_points, training_indices, spatial_cols_size):
        print(Data.shape)
        print(n)
        print(num_train_points)
        complete_D_train = np.zeros(
            [num_train_points, self.spatial_radius * n])
        # STICC data stack
        for i in range(num_train_points):
            for k in range(self.spatial_radius):
                if k == 0:
                    #complete_D_train[i][k * n:(k + 1) * n] = Data[i][1:(n + 1)]
                    complete_D_train[i][k * n:(k + 1) * n] = Data[i][:n]
                else:
                    if np.isnan(Data[i][n + k]):
                        complete_D_train[i][k * n:(k + 1) * n] = np.nan
                    else:
                        #complete_D_train[i][k * n:(k + 1) * n] = Data[int(Data[i][n + k])][1:(n + 1)]
                        complete_D_train[i][k * n:(k + 1) * n] = Data[int(Data[i][n + k])][:n]
        return complete_D_train

    def prepare_out_directory(self):
        str_NULL = self.prefix_string
        if not os.path.exists(os.path.dirname(str_NULL)):
            try:
                os.makedirs(os.path.dirname(str_NULL))
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise

        return str_NULL

    def load_data(self, input_file, dropna=False):
        #Data = np.loadtxt(input_file, delimiter=",")
        Data = np.genfromtxt(input_file, delimiter=",", missing_values='', filling_values=np.nan)
        if dropna:
            Data = Data[~np.isnan(Data).any(axis=1)]
        (m, n) = Data.shape  # m: num of observations, n: size of observation vector
        print("completed getting the data")
        return Data, m, n

    def log_parameters(self):
        print("lam_sparse", self.lambda_parameter)
        print("switch_penalty", self.switch_penalty)
        print("num_cluster", self.number_of_clusters)
        print("num stacked", self.spatial_radius)

    def predict_clusters(self, test_data=None):
        '''
        Given the current trained model, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            test_data = self.trained_model['complete_D_train']

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['cluster_mean_info'],
                                                         self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         test_data,
                                                         self.trained_model['spatial_series_col_size'])

        print(lle_all_points_clusters)

        # Update cluster points - using NEW smoothening
        clustered_points = updateClusters(lle_all_points_clusters, switch_penalty=self.switch_penalty, spatial_series_index=self.spatial_series_index,
                                          spatial_series_closest=self.spatial_series_closest, spatial_radius=self.spatial_radius)

        return(clustered_points)
