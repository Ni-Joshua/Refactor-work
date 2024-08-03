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
        self.variances = clean_means_variances['covariance']
        self.means = clean_means_variances['means']
        self.attributes = []
        for i in range(0, len(self.means[0])):
            attributename = "attr"+str(i+1)
            self.attributes.append(attributename)
        self.neighborcount = neighborcount
        self.rate = rate
        self.completeClean_data = []

        self.clean_data = []
        self.missing_data = []
        self.extreme_data = []
        self.ood_data = []
        self.synthetic_data = self.synthetic_data_info.drop(["CID", "geometry"], axis=1)
        self.synthetic_data = self.synthetic_data.dropna()
        self.synthetic_data["x"] = self.synthetic_data["x"] - np.mean(self.synthetic_data['x'])
        self.synthetic_data["y"] = self.synthetic_data["y"] - np.mean(self.synthetic_data['y'])
        self.synthetic_data["geometry"] = gpd.points_from_xy(x=self.synthetic_data.x, y=self.synthetic_data.y)

    def generate_attr(self, groups, means,variances):
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

        synthetic_data[self.attributes] = self.generate_attr(synthetic_data['clus_group'], self.means, self.variances)
        
        nearest_pt, n_pt_cols = self.get_neighbors(synthetic_data)
        synthetic_data = synthetic_data.join(nearest_pt)
        self.clean_data = np.array(synthetic_data[self.attributes])
        self.completeClean_data = synthetic_data

    def generate_defective(self):
        clusters = list(set(self.completeClean_data['clus_group']))
        for cluster in clusters:
            mu = self.means[cluster-1]
            Sigma = self.variances[cluster-1]
            clusterattrvals = self.completeClean_data.iloc[self.completeClean_data.groupby('clus_group').indices.get(cluster)][self.attributes]
            clusterattrvals = np.array(clusterattrvals)
            Npoints, attrcount = clusterattrvals.shape

            mask = np.random.choice([True, False], (Npoints, attrcount), p=[self.rate, 1-self.rate])

            X_missing = copy.deepcopy(clusterattrvals)
            X_missing[mask] = np.nan

            X_extreme = copy.deepcopy(clusterattrvals)
            defects = 3 * np.diagonal(Sigma).reshape((1,-1)) * np.random.choice([1,-1], (Npoints,attrcount)) + np.random.multivariate_normal(mean=mu, cov=Sigma)
            X_extreme[mask] = defects[mask]

            X_ood = copy.deepcopy(clusterattrvals)
            defects = mu.reshape((1,-1)) * poisson.rvs(1, size=(Npoints, attrcount))
            X_ood[mask] = defects[mask]


            self.missing_data.extend(X_missing)
            self.extreme_data.extend(X_extreme)
            self.ood_data.extend(X_ood)
        self.missing_data = np.array(self.missing_data)
        self.extreme_data = np.array(self.extreme_data)
        self.ood_data = np.array(self.ood_data)
    
    def generate_all(self):
        self.generate_clean()
        self.generate_defective()
        np.savez("All-synthetic-data.npz", gt= self.synthetic_data['clus_group'], clean_data = self.clean_data, missing_data = self.missing_data, extreme_data=self.extreme_data, ood_data = self.ood_data, position = np.array([self.synthetic_data['x'], self.synthetic_data['y']]))
        # self.completeClean_data.to_csv(r'synthetic_data_clean.txt', header=None, index=True, sep=',')

    def display_gt(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.scatter(self.synthetic_data['x'], self.synthetic_data['y'], c=self.synthetic_data['clus_group'], cmap="Set1")
        ax.set_axis_off()
        ax.title.set_text('Ground Truth')
        plt.show()

    def display_clean(self):
        for i in range(0, len(self.attributes)):
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.scatter(self.synthetic_data['x'], self.synthetic_data['y'], c=self.clean_data[:, i], cmap="Set1")
            ax.set_axis_off()
            ax.title.set_text('Clean Data, attr ' + str(i+1))
        plt.show()

    def display_missing(self):
        for i in range(0, len(self.attributes)):
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.scatter(self.synthetic_data['x'], self.synthetic_data['y'], c=self.missing_data[:, i], cmap="Set1")
            ax.set_axis_off()
            ax.title.set_text('Missing Defective Data, attr ' + str(i+1))
        plt.show()

    def display_extreme(self):
        for i in range(0, len(self.attributes)):
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.scatter(self.synthetic_data['x'], self.synthetic_data['y'], c=self.extreme_data[:, i], cmap="Set1")
            ax.set_axis_off()
            ax.title.set_text('Extreme Defective Data, attr ' + str(i+1))
        plt.show()
    
    def display_ood(self):
        for i in range(0, len(self.attributes)):
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.scatter(self.synthetic_data['x'], self.synthetic_data['y'], c=self.ood_data[:, i], cmap="Set1")
            ax.set_axis_off()
            ax.title.set_text('OOD Defective Data, attr ' + str(i+1))
        plt.show()
    
    def display_all(self):
        self.display_clean()
        self.display_missing()
        self.display_extreme()
        self.display_ood()



# temp = Synthesizer()
# temp.generate_all()
# temp.display_all()