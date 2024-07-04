import random
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import esda
import libpysal.weights as weights
from esda.moran import Moran
from esda.moran import Moran_Local
from shapely.geometry import Point, MultiPoint, LineString, Polygon, shape
import json
import pylab
import networkx as nx
import libpysal
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from collections import Counter
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.birch import birch
from pyclustering.cluster.cure import cure
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.optics import optics
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import copy

def save_data_for_sticc(data, knn, coord, filename):
    o = open(filename, "w")
    #o.write("attr1,attr2,attr3,attr4,attr5,n_pt_0,n_pt_1,n_pt_2")
    for i in range(data.shape[0]):
        feat_str = ",".join([str(j) for j in data[i]]).replace("nan","")
        nbs = ",".join([str(j) for j in knn[i]])
        locs = ",".join([str(j) for j in coord[i]])
        o.write("{},{},{}\n".format(feat_str, nbs, locs))
    o.close()

def load_data(defect_rate, idx):
    data = np.load("defective_data/synthetic-data-r-{:.2f}-{:d}.npz".format(defect_rate, idx))
    
    ground_truth = data["gt"]
    clean_features = data["clean"]
    missing_mask = data["missing"]
    extreme_features = data["extreme"]
    ood_features = data["ood"]
    
    print(np.sum(missing_mask) / (missing_mask.shape[0] * missing_mask.shape[1]))
    
    missing_features = copy.deepcopy(clean_features)
    missing_features[missing_mask==1] = np.nan
    
    coord = ground_truth[:,:2]
    
    pts_all = []
    for x, y in ground_truth[:,:2]:
        pts_all.append((x, y))
    kd = libpysal.cg.KDTree(np.array(pts_all))
    wnn = libpysal.weights.KNN(kd, 30)
    
    nearest_pt = pd.DataFrame().from_dict(wnn.neighbors, orient="index")
    nearest_pt = nearest_pt.to_numpy()
    
    #dropped_features = copy.deepcopy(clean_features)
    #dropped_features = dropped_features[(~np.isnan(missing_features)).all(axis=1)]
    
    #dropped_coord = (ground_truth[:,:2])[(~np.isnan(missing_features)).all(axis=1)]
    
    #pts_all = []
    #for x, y in dropped_coord:
    #    pts_all.append((x, y))
    #kd = libpysal.cg.KDTree(np.array(pts_all))
    #wnn = libpysal.weights.KNN(kd, 30)
    
    #dropped_nearest_pt = pd.DataFrame().from_dict(wnn.neighbors, orient="index")
    #dropped_nearest_pt = dropped_nearest_pt.to_numpy()
    
    save_data_for_sticc(clean_features, nearest_pt, coord, "synthetic-data-clean-{:.2f}-{:d}.txt".format(defect_rate, idx))
    save_data_for_sticc(missing_features, nearest_pt, coord, "synthetic-data-missing-{:.2f}-{:d}.txt".format(defect_rate, idx))
    save_data_for_sticc(extreme_features, nearest_pt, coord, "synthetic-data-extreme-{:.2f}-{:d}.txt".format(defect_rate, idx))
    save_data_for_sticc(ood_features, nearest_pt, coord, "synthetic-data-ood-{:.2f}-{:d}.txt".format(defect_rate, idx))
    #save_data_for_sticc(dropped_features, dropped_nearest_pt, dropped_coord, "synthetic-data-dropped-{:.2f}.txt".format(defect_rate))


#Sanity Check
data = np.load("defective_data/synthetic-data-clean.npz")
ground_truth = data["gt"]
clean_features = data["clean"]
clustering = DBSCAN(eps=2.1, min_samples=10).fit(clean_features)

Counter(clustering.labels_)
print("ARI", adjusted_rand_score(clustering.labels_, ground_truth[:,2]))

fig, ax = plt.subplots(figsize=(10, 5))
markersize = 10
plt.scatter(ground_truth[:,0], ground_truth[:,1], c=clustering.labels_, cmap="Set1")
ax.set_axis_off()
ax.title.set_text('Ground Truth')

pts_all = []
for x, y in ground_truth[:,:2]:
    pts_all.append((x, y))
kd = libpysal.cg.KDTree(np.array(pts_all))
wnn = libpysal.weights.KNN(kd, 30)

nearest_pt = pd.DataFrame().from_dict(wnn.neighbors, orient="index")
for i in range(nearest_pt.shape[1]):
    nearest_pt = nearest_pt.rename({i:f"n_pt_{i}"}, axis=1)
nearest_pt.head(1)

nearest_pt = nearest_pt.to_numpy()

save_data_for_sticc(clean_features, nearest_pt, "synthetic-data-clean.txt")

!rm -rf result-synthetic-data-clean-glasso.txt
!python STICC_main.py --fname=synthetic-data-clean.txt --oname=result-synthetic-data-clean-glasso.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method glasso
get_sticc_result(ground_truth, 'result-synthetic-data-clean-glasso.txt', title='STICC')
clean_features

kmeans = KMeans(n_clusters=7, n_init=300).fit(clean_features)
fig, ax = plt.subplots(figsize=(10, 5))
markersize = 10
plt.scatter(ground_truth[:,0], ground_truth[:,1], c=kmeans.labels_, cmap="Set1")
ax.set_axis_off()
ax.title.set_text('Ground Truth')
Counter(kmeans.labels_)
print("ARI", adjusted_rand_score(kmeans.labels_, ground_truth[:,2]))


#Load Data
defect_rate = "0.30"
data = np.load("defective_data/synthetic-data-r-{}.npz".format(defect_rate))
ground_truth = data["gt"]
clean_features = data["clean"]
missing_mask = data["missing"]
extreme_features = data["extreme"]
ood_features = data["ood"]
Counter(np.sum(missing_mask, axis=1))
print(np.sum(missing_mask) / (missing_mask.shape[0] * missing_mask.shape[1]))
missing_features = copy.deepcopy(clean_features)
missing_features[missing_mask==1] = np.nan
nidx = (~np.isnan(missing_features)).all(axis=1)
fig, ax = plt.subplots(figsize=(10, 5))
markersize = 10
plt.scatter(ground_truth[:,0], ground_truth[:,1], c=ground_truth[:,2], cmap="Set1")
ax.set_axis_off()
ax.title.set_text('Ground Truth')
pts_all = []
for x, y in ground_truth[:,:2]:
    pts_all.append((x, y))
kd = libpysal.cg.KDTree(np.array(pts_all))
wnn = libpysal.weights.KNN(kd, 30)
nearest_pt = pd.DataFrame().from_dict(wnn.neighbors, orient="index")
for i in range(nearest_pt.shape[1]):
    nearest_pt = nearest_pt.rename({i:f"n_pt_{i}"}, axis=1)
nearest_pt.head(1)
nearest_pt = nearest_pt.to_numpy()
save_data_for_sticc(clean_features, nearest_pt, "synthetic-data-clean-{}.txt".format(defect_rate))
save_data_for_sticc(missing_features, nearest_pt, "synthetic-data-missing-{}.txt".format(defect_rate))
save_data_for_sticc(extreme_features, nearest_pt, "synthetic-data-extreme-{}.txt".format(defect_rate))
save_data_for_sticc(ood_features, nearest_pt, "synthetic-data-ood-{}.txt".format(defect_rate))
print(np.sum(np.abs(extreme_features - clean_features)))
print(np.sum(np.abs(ood_features - clean_features)))

#STICC
def get_sticc_result(ground_truth, filename, title='STICC'):
    pred_labels = pd.read_table(filename, names=["group"])
    
#     fig, ax = plt.subplots(figsize=(10, 5))
#     markersize = 10
#     plt.scatter(ground_truth[:,0], ground_truth[:,1], c=pred_labels.group, cmap="Set1")
#     ax.set_axis_off()
#     ax.title.set_text('Ground Truth')

#     print("ARI", adjusted_rand_score(pred_labels.group, ground_truth[:,2]))
#     print("NMI", normalized_mutual_info_score(pred_labels.group, ground_truth[:,2]))
    
    return adjusted_rand_score(pred_labels.group, ground_truth[:,2]), normalized_mutual_info_score(pred_labels.group, ground_truth[:,2])
    
    #sp_contiguity = cal_joint_statistic(synthetic_data_sticc, w_voronoi)
    #print("Spatial contiguity: ", sp_contiguity)
    
    #get_max_f1_score(synthetic_data_sticc)
    
    #return synthetic_data_sticc

#Clean Values
!rm -rf result-synthetic-data-clean-glasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-clean-0.30.txt --oname=result-synthetic-data-clean-glasso-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=14 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method glasso
get_sticc_result(ground_truth, 'result-synthetic-data-clean-glasso-0.30.txt', title='STICC')

!rm -rf result-synthetic-data-clean-missglasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-clean-0.30.txt --oname=result-synthetic-data-clean-missglasso-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=14 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_random --local_radius=3 --init_random --defect_rate 0.2
get_sticc_result(ground_truth, 'result-synthetic-data-clean-missglasso-0.30.txt', title='STICC')
kmeans = KMeans(n_clusters=7, n_init=300).fit(clean_features)
fig, ax = plt.subplots(figsize=(10, 5))
markersize = 10
plt.scatter(ground_truth[:,0], ground_truth[:,1], c=kmeans.labels_, cmap="Set1")
ax.set_axis_off()
ax.title.set_text('Ground Truth')

print("ARI", adjusted_rand_score(kmeans.labels_, ground_truth[:,2]))
print("NMI", normalized_mutual_info_score(kmeans.labels_, ground_truth[:,2]))
!rm -rf result-synthetic-data-missing-missglasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-dropped-0.30.txt --oname=result-synthetic-data-missing-dropping-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method glasso

pred_labels = pd.read_table('result-synthetic-data-missing-dropping-0.30.txt', names=["group"])

print("ARI", adjusted_rand_score(pred_labels.group, (ground_truth[nidx])[:,2]))
!rm -rf result-synthetic-data-missing-missglasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-missing-0.30.txt --oname=result-synthetic-data-missing-missglasso-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso --local_radius 3
get_sticc_result(ground_truth, 'result-synthetic-data-missing-missglasso-0.30.txt', title='STICC')
!rm -rf result-synthetic-data-missing-missglasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-missing-0.30.txt --oname=result-synthetic-data-missing-missglasso-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso --local_radius 3
get_sticc_result(ground_truth, 'result-synthetic-data-missing-missglasso-0.30.txt', title='STICC')
get_sticc_result(ground_truth, 'result-synthetic-data-missing-missglasso-0.30.txt', title='STICC')

#Extreme Values
kmeans = KMeans(n_clusters=7, n_init=300).fit(extreme_features)
fig, ax = plt.subplots(figsize=(10, 5))
markersize = 10
plt.scatter(ground_truth[:,0], ground_truth[:,1], c=kmeans.labels_, cmap="Set1")
ax.set_axis_off()
ax.title.set_text('Ground Truth')

print("ARI", adjusted_rand_score(kmeans.labels_, ground_truth[:,2]))
Counter(kmeans.labels_)
!rm -rf result-synthetic-data-extreme-glasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-extreme-0.30.txt --oname=result-synthetic-data-extreme-glasso-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=14 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method glasso