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
                 clean_means_variances = pd.read_csv("clean-means-variances.csv"),
                 neighborcount=3, rate = 0.3):
        np.random.seed(1234)
        self.synthetic_data_info = synthetic_data_info
        self.clean_means_variances = clean_means_variances
        self.neighborcount = neighborcount
        self.rate = rate
        self.clean_data = []

    def get_attr(self, x, means,variances):
        return np.random.normal(means[x-1], variances[x-1])

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
        synthetic_data = self.synthetic_data_info.drop(["CID", "geometry"], axis=1)
        synthetic_data["x"] = synthetic_data["x"] + 9965410
        synthetic_data["y"] = synthetic_data["y"] - 5308400
        synthetic_data["geometry"] = gpd.points_from_xy(x=synthetic_data.x, y=synthetic_data.y)

        attributes = []
        for i in range(0, len(self.clean_means_variances.columns), 2):
            attributename = self.clean_means_variances.columns[i].split("_")[0]
            attributes.append(attributename)
            synthetic_data[attributename] = synthetic_data['clus_group'].apply(lambda x: self.get_attr(x, self.clean_means_variances.iloc[:,i], self.clean_means_variances.iloc[:,i+1] ))
        nearest_pt, n_pt_cols = self.get_neighbors(synthetic_data)
        synthetic_data = synthetic_data.join(nearest_pt)
        complete_data = synthetic_data[attributes+n_pt_cols]
        self.clean_data = complete_data
        complete_data.to_csv(r'synthetic_data_clean.csv', header=None, index=True, sep=',')

    def generate_defective(self):
        print("WIP")

x = Synthesizer()
x.generate_clean()
