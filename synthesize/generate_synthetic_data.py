import random
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
# import esda
import libpysal.weights as weights
# from esda.moran import Moran
# from esda.moran import Moran_Local
from shapely.geometry import Point, MultiPoint, LineString, Polygon, shape
import json
import pylab
import networkx as nx
import libpysal
import numpy as np
import scipy
from scipy.stats import multivariate_normal, poisson

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from utils import set_seed, create_random_mask, mean_imputation, compute_mse, plot_solution

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.birch import birch
from pyclustering.cluster.cure import cure
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.optics import optics
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import copy

class Synthesizer:
    def __init__(self, synthetic_data_info = gpd.read_file('sticc_points.shp'), 
                 clean_means_variances = np.load("clean-means-variances.npz"),
                 neighborcount=3, rate = 0.3):
        np.random.seed(1234)
        self.synthetic_data_info = synthetic_data_info
        self.clean_means_variances = clean_means_variances
        self.neighborcount = neighborcount
        self.rate = rate
        self.clean_data = None
        self.synthetic_data = self.synthetic_data_info.drop(["CID", "geometry"], axis=1)
        self.synthetic_data = self.synthetic_data.dropna()
        self.synthetic_data["x"] = self.synthetic_data["x"] - np.mean(self.synthetic_data['x'])
        self.synthetic_data["y"] = self.synthetic_data["y"] - np.mean(self.synthetic_data['y'])
        self.synthetic_data["geometry"] = gpd.points_from_xy(x=self.synthetic_data.x, y=self.synthetic_data.y)

    def get_attr(self, groups, means,variances):
        attrvalues = []
        for index in range(0, len(groups)):
            attrvalues.append(np.random.multivariate_normal(means[groups[index]-1], variances[groups[index]-1]))
        return attrvalues
    
    def get_neighbors(self, synthetic_data):
        pts_all = []
        for pt in synthetic_data.iterrows():
            pts_all.append((pt[1].x, pt[1].y))
        kd = libpysal.cg.KDTree(np.array(pts_all))
        wnn = libpysal.weights.KNN(kd, self.neighborcount)

        nearest_pt = pd.DataFrame().from_dict(wnn.neighbors, orient="index") 
        n_pt_cols = []
        for i in range(nearest_pt.shape[1]):
            name = f"n_pt_{i}"
            n_pt_cols.append(name)
            nearest_pt = nearest_pt.rename({i:name}, axis=1)
        return nearest_pt, n_pt_cols
    
    def generate_clean(self):
        synthetic_data = self.synthetic_data
        attributes = []
        variances = self.clean_means_variances['covariance']
        means = self.clean_means_variances['means']
        for i in range(0, len(means[0])):
            attributename = "attr"+str(i+1)
            attributes.append(attributename)
        synthetic_data[attributes] = self.get_attr(synthetic_data['clus_group'], means, variances)
        
        nearest_pt, n_pt_cols = self.get_neighbors(synthetic_data)
        synthetic_data = synthetic_data.join(nearest_pt)
        complete_data = synthetic_data[attributes+n_pt_cols]
        self.clean_data = synthetic_data
        complete_data.to_csv(r'synthetic_data_clean.txt', header=None, index=True, sep=',')

    def generate_defective(self, X, mu, Sigma, rate=0.1):
        N, d = X.shape
        mask = np.random.choice([True, False], (N, d), p=[rate, 1-rate])

        if mode == "missing":
            X_missing = copy.deepcopy(X)
            X_missing[mask] = np.nan
        elif mode == "extreme":
            X_extreme = copy.deepcopy(X)
            defects = 3 * np.diagonal(Sigma).reshape((1,-1)) * np.random.choice([1,-1], (N,d)) + multivariate_normal(mean=mu, cov=Sigma)
            X_extreme[mask] = defects[mask]
        elif mode == "ood":
            X_ood = copy.deepcopy(X)
            defects = mu.reshape((1,-1)) * poisson.rvs(1, size=(N, d))
            X_ood[mask] = defects[mask]
        else:
            assert False, "Unknown defect mode!"

        return X_missing, X_extreme, X_ood

    def display_gt(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.scatter(self.synthetic_data['x'], self.synthetic_data['y'], c=self.synthetic_data['clus_group'], cmap="Set1")
        ax.set_axis_off()
        ax.title.set_text('Ground Truth')
        plt.show()
