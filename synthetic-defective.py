import random
import os
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
from sklearn.tree import DecisionTreeRegressor
from collections import Counter
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.birch import birch
from pyclustering.cluster.cure import cure
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.optics import optics
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import copy

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

import scipy.stats as st

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

def printvalues(mari, confari, mnmi, confnmi):
    print("{:.2f} $\pm$ {:.2f}".format(100*mari, 100*(confari[1] - mari)))
    print("{:.2f} $\pm$ {:.2f}".format(100*mnmi, 100*(confnmi[1] - mnmi)))

#Sanity Check
data = np.load("defective_data/synthetic-data-clean.npz")
ground_truth = data["gt"]
clean_features = data["clean"]
clustering = DBSCAN(eps=2.1, min_samples=10).fit(clean_features)

# Counter(clustering.labels_)
# print("ARI", adjusted_rand_score(clustering.labels_, ground_truth[:,2]))

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
get_sticc_result(ground_truth, 'result-synthetic-data-extreme-glasso-0.30.txt', title='STICC')
!rm -rf result-synthetic-data-extreme-missglasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-extreme-0.30.txt --oname=result-synthetic-data-extreme-missglasso-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_random --defect_rate 0.3
get_sticc_result(ground_truth, 'result-synthetic-data-extreme-missglasso-0.30.txt', title='STICC')

#OOD values
kmeans = KMeans(n_clusters=7, n_init=300).fit(ood_features)
fig, ax = plt.subplots(figsize=(10, 5))
markersize = 10
plt.scatter(ground_truth[:,0], ground_truth[:,1], c=kmeans.labels_, cmap="Set1")
ax.set_axis_off()
ax.title.set_text('Ground Truth')

print("ARI", adjusted_rand_score(kmeans.labels_, ground_truth[:,2]))
Counter(kmeans.labels_)

!rm -rf result-synthetic-data-ood-glasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-ood-0.30.txt --oname=result-synthetic-data-ood-glasso-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=14 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method glasso 
get_sticc_result(ground_truth, 'result-synthetic-data-ood-glasso-0.30.txt', title='STICC')
!rm -rf result-synthetic-data-ood-missglasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-ood-0.30.txt --oname=result-synthetic-data-ood-missglasso-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_random --defect_rate 0.10
get_sticc_result(ground_truth, 'result-synthetic-data-ood-missglasso-0.30.txt', title='STICC')
!rm -rf result-synthetic-data-ood-missglasso-init-0.30.txt
!python STICC_main.py --fname=synthetic-data-ood-0.30.txt --oname=result-synthetic-data-ood-missglasso-init-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_random --defect_rate 0.10 --init_random
get_sticc_result(ground_truth, 'result-synthetic-data-ood-missglasso-init-0.30.txt', title='STICC')
!rm -rf result-synthetic-data-ood-missglasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-ood-0.30.txt --oname=result-synthetic-data-ood-missglasso-globalmean-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_random --defect_rate 0.10 --init_random --interp global_mean
get_sticc_result(ground_truth, 'result-synthetic-data-ood-missglasso-globalmean-0.30.txt', title='STICC')
!rm -rf result-synthetic-data-ood-missglasso-0.30.txt
!python STICC_main.py --fname=synthetic-data-ood-0.30.txt --oname=result-synthetic-data-ood-missglasso-localmean-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_random --defect_rate 0.10 --init_random --interp local_mean --local_radius 10
get_sticc_result(ground_truth, 'result-synthetic-data-ood-missglasso-localmean-0.30.txt', title='STICC')

#Global Mean Interpolation
!rm -rf result-synthetic-data-missing-globalmean-0.30.txt
!python STICC_main.py --fname=synthetic-data-missing-0.30.txt --oname=result-synthetic-data-missing-globalmean-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method global_mean
get_sticc_result(ground_truth, 'result-synthetic-data-missing-globalmean-0.30.txt', title='STICC')

#Local Mean Interpolation
!rm -rf result-synthetic-data-missing-localmean-0.30.txt
!python STICC_main.py --fname=synthetic-data-missing-0.30.txt --oname=result-synthetic-data-missing-localmean-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method local_mean --local_radius 3
get_sticc_result(ground_truth, 'result-synthetic-data-missing-localmean-0.30.txt', title='STICC')

#Kriging Interpolation
!rm -rf result-synthetic-data-missing-kriging-0.30.txt
!python STICC_main.py --fname=synthetic-data-missing-0.30.txt --oname=result-synthetic-data-missing-kriging-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 --coord_idx_start=35 --coord_idx_end=36\
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method kriging
get_sticc_result(ground_truth, 'result-synthetic-data-missing-kriging-0.30.txt', title='STICC')

#DT Interpolation
nidxs = np.isnan(missing_features)
tree_data = np.hstack((ground_truth[:,:2], clean_features))
X = tree_data[(~nidxs).all(axis=1),:5]
y = tree_data[(~nidxs).all(axis=1),5:]
X.shape
X_test = tree_data[nidxs.any(axis=1),:5]
y_test = tree_data[nidxs.any(axis=1),5:]
X_test.shape
regr = DecisionTreeRegressor(max_depth=8)
regr.fit(X, y)
y_pred = regr.predict(X_test)
X_test
y_pred
y_test
tree_data[:,np.array([1,0,0,1,1,0,0]).astype(bool)]
tree_data[:,~np.array([True,False,False,True,True,False,False])]
int(5/2)
!rm -rf result-synthetic-data-missing-tree-0.30.txt
!python STICC_main.py --fname=synthetic-data-missing-0.30.txt --oname=result-synthetic-data-missing-tree-0.30.txt --attr_idx_start=0 \
--attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 --coord_idx_start=35 --coord_idx_end=36\
--spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method tree --local_radius 3
get_sticc_result(ground_truth, 'result-synthetic-data-missing-tree-0.30.txt', title='STICC')

#Batch Experiment
for i in range(10):
    for r in np.arange(0.05,0.55,0.05):
        load_data(r, i)

def batch_experiment_clean_kmeans():
    aris, nmis = [], []
    for i in range(10):
        data = np.load("defective_data/synthetic-data-r-0.05-{:d}.npz".format(i))
        ground_truth = data["gt"]
        clean_features = data["clean"]
        kmeans = KMeans(n_clusters=7, n_init=300).fit(clean_features)
        ari, nmi = adjusted_rand_score(kmeans.labels_, ground_truth[:,2]), normalized_mutual_info_score(kmeans.labels_, ground_truth[:,2])
        aris.append(ari)
        nmis.append(nmi)
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi

def batch_experiment_clean_sticc():
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    for i in range(10):
        os.system("""python STICC_main.py --fname=synthetic-data-clean-0.05-{}.txt --oname=result-synthetic-data-clean-glasso-0.05-{}.txt --attr_idx_start=0 \
        --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
        --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method glasso""".format(i, i))
        ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-clean-glasso-0.05-{}.txt".format(i))
        aris.append(ari)
        nmis.append(nmi)
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi

def batch_experiment_clean_stable_sticc(radius, rate):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    for i in range(10):
        os.system("""python STICC_main.py --fname=synthetic-data-clean-0.05-{}.txt --oname=result-synthetic-data-clean-missglasso-0.05-{}-{}-{}.txt --attr_idx_start=0 \
        --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
        --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_random --local_radius {} --defect_rate {} --init_random""".format(i, i, radius, rate, radius, rate))
        ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-clean-missglasso-0.05-{}-{}-{}.txt".format(i, radius, rate))
        aris.append(ari)
        nmis.append(nmi)
        
        print("Finished batch {}".format(i))
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi
def batch_experiment_missing_sticc_global_mean(rate):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    nc = 0
    for i in range(10):
        try:
            os.system("""python STICC_main.py --fname=synthetic-data-missing-{}-{}.txt --oname=result-synthetic-data-missing-globalmean-{}-{}.txt --attr_idx_start=0 \
            --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
            --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method global_mean""".format(rate, i, rate, i))
            ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-missing-globalmean-{}-{}.txt".format(rate, i))
            aris.append(ari)
            nmis.append(nmi)
        except:
            nc += 1
        
        print("Finished batch {}".format(i))
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi, nc

def batch_experiment_missing_sticc_local_mean(rate):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    nc = 0
    for i in range(10):
        try:
            os.system("""python STICC_main.py --fname=synthetic-data-missing-{}-{}.txt --oname=result-synthetic-data-missing-globalmean-{}-{}.txt --attr_idx_start=0 \
            --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
            --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method local_mean --local_radius 3""".format(rate, i, rate, i))
            ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-missing-globalmean-{}-{}.txt".format(rate, i))
            aris.append(ari)
            nmis.append(nmi)
        except:
            nc += 1
        
        print("Finished batch {}".format(i))
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi, nc

def batch_experiment_missing_sticc_tree(rate):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    nc = 0
    for i in range(10):
        try:
            os.system("""python STICC_main.py --fname=synthetic-data-missing-{}-{}.txt --oname=result-synthetic-data-missing-tree-{}-{}-depth-5.txt --attr_idx_start=0 \
            --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
            --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method tree --local_radius 3""".format(rate, i, rate, i))
            ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-missing-tree-{}-{}-depth-5.txt".format(rate, i))
            aris.append(ari)
            nmis.append(nmi)
        except:
            nc += 1
        
        print("Finished batch {}".format(i))
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi, nc

def batch_experiment_missing_sticc_kriging(rate):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    nc = 0
    for i in range(10):
        try:
            os.system("""python STICC_main.py --fname=synthetic-data-missing-{}-{}.txt --oname=result-synthetic-data-missing-kriging-{}-{}.txt --attr_idx_start=0 \
            --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
            --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method kriging""".format(rate, i, rate, i))
            ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-missing-kriging-{}-{}.txt".format(rate, i))
            aris.append(ari)
            nmis.append(nmi)
        except:
            nc += 1
        
        print("Finished batch {}".format(i))
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi, nc

def batch_experiment_missing_stable_sticc(rate, radius):
    nc = 0
    for i in range(10):
        try:
            os.system("""python STICC_main.py --fname=synthetic-data-missing-{}-{}.txt --oname=result-synthetic-data-missing-missglasso-{}-{}-{}.txt --attr_idx_start=0 \
            --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
            --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso --local_radius {}""".format(rate, i, rate, i, radius, radius))
        except:
            nc += 1
        
        print("Finished batch {}".format(i))
    
    return nc

results_clean_kmeans = batch_experiment_clean_kmeans()
print("{:.4f} $\pm$ {:.4f}".format(results_clean_kmeans[2], results_clean_kmeans[4][1] - results_clean_kmeans[2]))
print("{:.4f} $\pm$ {:.4f}".format(results_clean_kmeans[3], results_clean_kmeans[5][1] - results_clean_kmeans[3]))
results_clean_sticc = batch_experiment_clean_sticc()
print("{:.4f} $\pm$ {:.4f}".format(results_clean_sticc[2], results_clean_sticc[4][1] - results_clean_sticc[2]))
print("{:.4f} $\pm$ {:.4f}".format(results_clean_sticc[3], results_clean_sticc[5][1] - results_clean_sticc[3]))
results_clean_stable_sticc_10_005 = batch_experiment_clean_stable_sticc(radius=10, rate=0.05)
aris, nmis, mari, mnmi, confari, confnmi = results_clean_stable_sticc_10_005

print("{:.4f} $\pm$ {:.4f}".format(mari, confari[1] - mari))
print("{:.4f} $\pm$ {:.4f}".format(mnmi, confnmi[1] - mnmi))
print("{:.4f} $\pm$ {:.4f}".format(results_clean_stable_sticc[2], results_clean_stable_sticc[4][1] - results_clean_stable_sticc[2]))
print("{:.4f} $\pm$ {:.4f}".format(results_clean_stable_sticc[3], results_clean_stable_sticc[5][1] - results_clean_stable_sticc[3]))
results_clean_stable_sticc_10_01 = batch_experiment_clean_stable_sticc(radius=10, rate=0.1)
aris, nmis, mari, mnmi, confari, confnmi = results_clean_stable_sticc_10_01

print("{:.4f} $\pm$ {:.4f}".format(mari, confari[1] - mari))
print("{:.4f} $\pm$ {:.4f}".format(mnmi, confnmi[1] - mnmi))
results_clean_stable_sticc_10_02 = batch_experiment_clean_stable_sticc(radius=10, rate=0.2)

aris, nmis, mari, mnmi, confari, confnmi = results_clean_stable_sticc_10_02

print("{:.4f} $\pm$ {:.4f}".format(mari, confari[1] - mari))
print("{:.4f} $\pm$ {:.4f}".format(mnmi, confnmi[1] - mnmi))
results_missing_sticc_globalmean_030 = batch_experiment_missing_sticc_global_mean("0.30")

aris, nmis, mari, mnmi, confari, confnmi = results_clean_stable_sticc_globalmean_005

printvalues(mari, confari, mnmi, confnmi)

aris, nmis, mari, mnmi, confari, confnmi = results_missing_sticc_globalmean_010

printvalues(mari, confari, mnmi, confnmi)

aris, nmis, mari, mnmi, confari, confnmi, nc = results_missing_sticc_globalmean_030

printvalues(mari, confari, mnmi, confnmi)

results_missing_sticc_localmean_005 = batch_experiment_missing_sticc_local_mean("0.05")
aris, nmis, mari, mnmi, confari, confnmi, nc = results_missing_sticc_localmean_005

printvalues(mari, confari, mnmi, confnmi)
print(nc)
rate = "0.30"

aris, nmis = [], []
data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
ground_truth = data["gt"]

for i in range(10):
    ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-missing-localmean-{}-{}.txt".format(rate, i))
    aris.append(ari)
    nmis.append(nmi)
    
mari = np.mean(aris)
mnmi = np.mean(nmis)
confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
printvalues(mari, confari, mnmi, confnmi)

results_missing_sticc_tree_030_depth_5 = batch_experiment_missing_sticc_tree("0.30")
aris, nmis, mari, mnmi, confari, confnmi, nc = results_missing_sticc_tree_030_depth_5

printvalues(mari, confari, mnmi, confnmi)
print(nc)

aris, nmis, mari, mnmi, confari, confnmi, nc = results_missing_sticc_tree_005

printvalues(mari, confari, mnmi, confnmi)
print(nc)

aris, nmis, mari, mnmi, confari, confnmi, nc = results_missing_sticc_tree_010

printvalues(mari, confari, mnmi, confnmi)
print(nc)

aris, nmis, mari, mnmi, confari, confnmi, nc = results_missing_sticc_tree_030

printvalues(mari, confari, mnmi, confnmi)
print(nc)

aris

results_missing_sticc_kriging_030 = batch_experiment_missing_sticc_kriging("0.30")
results_missing_sticc_kriging_030_mean = batch_experiment_missing_sticc_kriging("0.30")

aris, nmis, mari, mnmi, confari, confnmi, nc = results_missing_sticc_kriging_030_mean

printvalues(mari, confari, mnmi, confnmi)
print(nc)

aris, nmis, mari, mnmi, confari, confnmi, nc = results_missing_sticc_kriging_030

printvalues(mari, confari, mnmi, confnmi)
print(nc)

rate = "0.30"

aris, nmis = [], []
data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
ground_truth = data["gt"]

for i in range(10):
    ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-missing-kriging-{}-{}.txt".format(rate, i))
    aris.append(ari)
    nmis.append(nmi)
    
mari = np.mean(aris)
mnmi = np.mean(nmis)
confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
printvalues(mari, confari, mnmi, confnmi)

aris
nc = batch_experiment_missing_stable_sticc("0.30", "5")

rate = "0.30"
radius = "5"

aris, nmis = [], []
data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
ground_truth = data["gt"]

for i in range(10):
    try:
        ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-missing-missglasso-{}-{}-{}.txt".format(rate, i, radius))
    except:
        continue
    aris.append(ari)
    nmis.append(nmi)
    
mari = np.mean(aris)
mnmi = np.mean(nmis)
confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
printvalues(mari, confari, mnmi, confnmi)

aris

aris = [0.9393839521639407,
 0.8891235990632819,
 0.9601949197256107,
 0.9485859008745027,
 0.9494810205445021,
 0.9502557438046737,
 0.9352221463500014,
 0.9579160055482231,
 0.7193183620612167,
 0.9392619987832969]

np.mean(aris)

nmis

#Batch Extreme experiment
def batch_experiment_extreme_kmeans(rate):
    aris, nmis = [], []
    for i in range(10):
        data = np.load("defective_data/synthetic-data-r-{}-{:d}.npz".format(rate, i))
        ground_truth = data["gt"]
        clean_features = data["extreme"]
        kmeans = KMeans(n_clusters=7, n_init=300).fit(clean_features)
        ari, nmi = adjusted_rand_score(kmeans.labels_, ground_truth[:,2]), normalized_mutual_info_score(kmeans.labels_, ground_truth[:,2])
        aris.append(ari)
        nmis.append(nmi)
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi
aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_extreme_kmeans("0.05")
printvalues(mari, confari, mnmi, confnmi)
aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_extreme_kmeans("0.10")
printvalues(mari, confari, mnmi, confnmi)
aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_extreme_kmeans("0.30")
printvalues(mari, confari, mnmi, confnmi)

def batch_experiment_extreme_sticc(rate):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    for i in range(10):
        os.system("""python STICC_main.py --fname=synthetic-data-extreme-{}-{}.txt --oname=result-synthetic-data-extreme-glasso-{}-{}.txt --attr_idx_start=0 \
        --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
        --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method glasso""".format(rate, i, rate, i))
        ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-extreme-glasso-{}-{}.txt".format(rate, i))
        aris.append(ari)
        nmis.append(nmi)
        
        print("Finished batch {}".format(i))
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi

aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_extreme_sticc("0.05")
printvalues(mari, confari, mnmi, confnmi)
aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_extreme_sticc("0.10")
printvalues(mari, confari, mnmi, confnmi)
aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_extreme_sticc("0.30")
printvalues(mari, confari, mnmi, confnmi)
def batch_experiment_extreme_stable_sticc(rate, radius, defect):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    for i in range(10):
        os.system("""python STICC_main.py --fname=synthetic-data-extreme-{}-{}.txt --oname=result-synthetic-data-extreme-missglasso-{}-{}-{}.txt --attr_idx_start=0 \
        --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
        --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_random --local_radius {} --defect_rate {} --init_random --interp missglasso""".format(rate, i, rate, i, defect, radius, defect))
        ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-extreme-missglasso-{}-{}-{}.txt".format(rate, i, defect))
        aris.append(ari)
        nmis.append(nmi)
        
        print("\n\n\nFinished batch {}".format(i))
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi
aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_extreme_stable_sticc("0.30", "10", "0.05")
def batch_experiment_extreme_localmean(rate, radius, defect):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    for i in range(10):
        os.system("""python STICC_main.py --fname=synthetic-data-extreme-{}-{}.txt --oname=result-synthetic-data-extreme-localmean-{}-{}-{}.txt --attr_idx_start=0 \
        --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 \
        --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_random --local_radius {} --defect_rate {} --init_random --interp localmean""".format(rate, i, rate, i, defect, radius, defect))
        ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-extreme-localmean-{}-{}-{}.txt".format(rate, i, defect))
        aris.append(ari)
        nmis.append(nmi)
        
        print("\n\n\nFinished batch {}".format(i))
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi
aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_extreme_stable_sticc("0.30", "10", "0.10")

rate = "0.30"
defect = "0.05"

aris, nmis = [], []
data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
ground_truth = data["gt"]
for i in range(10):
    ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-extreme-missglasso-{}-{}-{}.txt".format(rate, i, defect))
    aris.append(ari)
    nmis.append(nmi)
    
mari = np.mean(aris)
mnmi = np.mean(nmis)
confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))

printvalues(mari, confari, mnmi, confnmi)

printvalues(mari, confari, mnmi, confnmi)

printvalues(mari, confari, mnmi, confnmi)

printvalues(mari, confari, mnmi, confnmi)

aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_extreme_stable_sticc("0.05", "3", "0.10")

printvalues(mari, confari, mnmi, confnmi)

#OOD Batch Experiment
def batch_experiment_ood_kmeans(rate):
    aris, nmis = [], []
    for i in range(10):
        data = np.load("defective_data/synthetic-data-r-{}-{:d}.npz".format(rate, i))
        ground_truth = data["gt"]
        clean_features = data["ood"]
        kmeans = KMeans(n_clusters=7, n_init=300).fit(clean_features)
        ari, nmi = adjusted_rand_score(kmeans.labels_, ground_truth[:,2]), normalized_mutual_info_score(kmeans.labels_, ground_truth[:,2])
        aris.append(ari)
        nmis.append(nmi)
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi

aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_ood_kmeans(0.05)

printvalues(mari, confari, mnmi, confnmi)

aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_ood_kmeans("0.10")
printvalues(mari, confari, mnmi, confnmi)
aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_ood_kmeans("0.30")
printvalues(mari, confari, mnmi, confnmi)

def batch_experiment_ood_sticc(rate):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    nc = 0
    for i in range(10):
        try:
            os.system("rm -rf result-synthetic-data-ood-glasso-{}-{}.txt".format(rate, i))
            os.system("""python -W ignore STICC_main.py --fname=synthetic-data-ood-{}-{}.txt --oname=result-synthetic-data-ood-glasso-{}-{}.txt --attr_idx_start=0 \
            --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 --coord_idx_start 35 --coord_idx_end 36 \
            --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method glasso""".format(rate, i, rate, i))
            ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-ood-glasso-{}-{}.txt".format(rate, i))
            aris.append(ari)
            nmis.append(nmi)
        except:
            nc += 1
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi

aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_ood_sticc("0.05")

rate = "0.05"

aris, nmis = [], []
data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
ground_truth = data["gt"]
for i in range(10):
    try:
        ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-ood-glasso-{}-{}.txt".format(rate, i))
        aris.append(ari)
        nmis.append(nmi)
    except:
        pass
    
mari = np.mean(aris)
mnmi = np.mean(nmis)
confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))

printvalues(mari, confari, mnmi, confnmi)

def batch_experiment_ood_stable_sticc(rate, radius, thres):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    nc = 0
    for i in range(10):
        try:
            os.system("""python -W ignore STICC_main.py --fname=synthetic-data-ood-{}-{}.txt --oname=result-synthetic-data-ood-stable-static-localmean-{}-{}.txt --attr_idx_start=0 \
            --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34 --coord_idx_start 35 --coord_idx_end 36 \
            --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_static --interp local_mean --local_radius {} --mala_thres {}""".format(rate, i, rate, i, radius, thres))
            ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-ood-stable-static-localmean-{}-{}.txt".format(rate, i))
            aris.append(ari)
            nmis.append(nmi)
        except:
            nc += 1
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi

aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_ood_stable_sticc("0.05", "3", "9.236")

printvalues(mari, confari, mnmi, confnmi)

def batch_experiment_ood_stable_static_localmean(rate, radius, thres):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    nc = 0
    for i in range(10):
        try:
            os.system("rm -rf result-synthetic-data-ood-stable-static-localmean-{}-{}.txt".format(rate, i))
            os.system("""python -W ignore STICC_main.py --fname=synthetic-data-ood-{}-{}.txt --oname=result-synthetic-data-ood-stable-static-localmean-{}-{}.txt --attr_idx_start=0 \
            --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34  --coord_idx_start 35 --coord_idx_end 36\
            --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_static --interp local_mean --local_radius {} --mala_thres {}""".format(rate, i, rate, i, radius, thres))
            ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-ood-stable-static-localmean-{}-{}.txt".format(rate, i))
            aris.append(ari)
            nmis.append(nmi)
        except:
            nc += 1
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi

aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_ood_stable_static_localmean("0.05", "3", "5.68")

printvalues(mari, confari, mnmi, confnmi)

def batch_experiment_ood_stable_dynamic(rate, radius, thres, mask_rate):
    aris, nmis = [], []
    data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
    ground_truth = data["gt"]
    nc = 0
    for i in range(10):
        try:
            os.system("rm -rf result-synthetic-data-ood-stable-dynamic-{}-{}.txt".format(rate, i))
            os.system("""python -W ignore STICC_main.py --fname=synthetic-data-ood-{}-{}.txt --oname=result-synthetic-data-ood-stable-dynamic-{}-{}-{}.txt --attr_idx_start=0 \
            --attr_idx_end=4 --spatial_idx_start=5 --spatial_idx_end=34  --coord_idx_start 35 --coord_idx_end 36\
            --spatial_radius=3 --number_of_clusters 7 --lambda_parameter 0.01 --beta 3 --maxIters 20 --method missglasso_dynamic --local_radius {} --mala_thres {} --mask_rate {}""".format(rate, i, rate, i, mask_rate, radius, thres, mask_rate))
            ari, nmi = get_sticc_result(ground_truth, "result-synthetic-data-ood-stable-dynamic-{}-{}-{}.txt".format(rate, i, mask_rate))
            aris.append(ari)
            nmis.append(nmi)
        except:
            nc += 1
    
    mari = np.mean(aris)
    mnmi = np.mean(nmis)
    confari = st.t.interval(0.95, len(aris)-1, loc=np.mean(aris), scale=st.sem(aris))
    confnmi = st.t.interval(0.95, len(nmis)-1, loc=np.mean(nmis), scale=st.sem(nmis))
    
    return aris, nmis, mari, mnmi, confari, confnmi

aris, nmis, mari, mnmi, confari, confnmi = batch_experiment_ood_stable_dynamic("0.30", "3", "5.68", "0.10")

printvalues(mari, confari, mnmi, confnmi)