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

# from pyclustering.cluster import cluster_visualizer
# from pyclustering.cluster.birch import birch
# from pyclustering.cluster.cure import cure
# from pyclustering.cluster.dbscan import dbscan
# from pyclustering.cluster.kmeans import kmeans
# from pyclustering.cluster.optics import optics
# from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

def permutation(lst):
    # if len(lst) == 0:
    #     return []

    # if len(lst) == 1:
    #     return [lst]
    l = []
    for i in range(len(lst)):
        if(len(lst) == 1):
            l.append(lst)
        else:
            m = lst[i]
            remLst = lst[:i] + lst[i+1:]
            for p in permutation(remLst):
                l.append([m] + p)       
    return l

def get_f1_score(df, permut):
    def match_clus(x, permut):
        # if x == 0:
        #     return int(permut[0])
        # elif x == 1:
        #     return int(permut[1])
        # elif x == 2:
        #     return int(permut[2])
        # elif x == 3:
        #     return int(permut[3])
        # elif x == 4:
        #     return int(permut[4])
        # elif x == 5:
        #     return int(permut[5])
        # elif x == 6:
        #     return int(permut[6])
        if(x<=6):
            return int(permut(x))
        else:
            return x

    df["group_match"] = df["group"].apply(lambda x: match_clus(x, permut))
    return df, f1_score(df.group_match.values, df.clus_group_gt.values, average='macro')

def get_max_f1_score(df):
    max_f1 = 0
    max_p = []
    for p in permutation([1,2,3,4,5,6,7]):
        df, f1 = get_f1_score(df, p)
        if max_f1 < f1:
            max_f1 = f1
            max_p = p
    print("f1_score ", max_f1, max_p)

def cal_joint_statistic(synthetic_data_sticc, w_voronoi):
    matched_connects = 0
    all_neighbors_connects = 0
    for obj_id, neighbors in w_voronoi.neighbors.items():
        obj_clus = synthetic_data_sticc.iat[obj_id, -1]
        for nei in neighbors:
            nei_clus = synthetic_data_sticc.iat[nei, -1]
            all_neighbors_connects += 1
            if obj_clus == nei_clus:
                matched_connects += 1
    return matched_connects / all_neighbors_connects

#Ground Truth
synthetic_data = gpd.read_file('data/sticc_points.shp')
synthetic_data = synthetic_data.drop(["CID", "geometry"], axis=1)
synthetic_data["x"] = synthetic_data["x"] + 9965410
synthetic_data["y"] = synthetic_data["y"] - 5308400
synthetic_data["geometry"] = gpd.points_from_xy(x=synthetic_data.x, y=synthetic_data.y)
synthetic_data.head(1)

def get_gt(x):
    if x == 8:
        return 3
    elif x == 4:
        return 1
    elif x == 9:
        return 4
    elif x == 10:
        return 2
    else:
        return x
    
synthetic_data["clus_group_gt"] = synthetic_data["clus_group"].apply(lambda x: get_gt(x))
synthetic_data.head(1)

fig, ax = plt.subplots(figsize=(10, 5))
markersize = 10
synthetic_data.plot(ax=ax, column="clus_group_gt", cmap="Set1", markersize=markersize, legend=True)
ax.set_axis_off()
ax.title.set_text('Ground Truth')

np.random.seed(1234)
def get_attr1(x):
    if x == 1:
        return np.random.normal(4, 1)
    elif x == 2:
        return np.random.normal(5, 1)
    elif x == 3:
        return np.random.normal(6, 1)
    elif x == 4:
        return np.random.normal(1, 1)
    elif x == 5:
        return np.random.normal(3, 1)
    elif x == 6:
        return np.random.normal(7, 1)
    elif x == 7:
        return np.random.normal(2, 1)
    
synthetic_data["attr1"] = synthetic_data["clus_group_gt"].apply(lambda x: get_attr1(x))

def get_attr2(x):
    if x == 1:
        return np.random.normal(1, 3)
    elif x == 2:
        return np.random.normal(7, 3)
    elif x == 3:
        return np.random.normal(2, 3)
    elif x == 4:
        return np.random.normal(3, 3)
    elif x == 5:
        return np.random.normal(6, 3)
    elif x == 6:
        return np.random.normal(4, 3)
    elif x == 7:
        return np.random.normal(5, 3)
    
synthetic_data["attr2"] = synthetic_data["clus_group_gt"].apply(lambda x: get_attr2(x))\

def get_attr3(x):
    if x == 1:
        return np.random.normal(80, 20)
    elif x == 2:
        return np.random.normal(30, 20)
    elif x == 3:
        return np.random.normal(20, 20)
    elif x == 4:
        return np.random.normal(100, 20)
    elif x == 5:
        return np.random.normal(60, 20)
    elif x == 6:
        return np.random.normal(70, 20)
    elif x == 7:
        return np.random.normal(40, 20)
    
synthetic_data["attr3"] = synthetic_data["clus_group_gt"].apply(lambda x: get_attr3(x))

def get_attr4(x):
    if x == 1:
        return np.random.normal(1000, 350)
    elif x == 2:
        return np.random.normal(900, 350)
    elif x == 3:
        return np.random.normal(600, 350)
    elif x == 4:
        return np.random.normal(700, 350)
    elif x == 5:
        return np.random.normal(800, 350)
    elif x == 6:
        return np.random.normal(400, 350)
    elif x == 7:
        return np.random.normal(500, 350)
    
synthetic_data["attr4"] = synthetic_data["clus_group_gt"].apply(lambda x: get_attr4(x))

def get_attr5(x):
    if x == 1:
        return np.random.normal(999, 3)
    elif x == 2:
        return np.random.normal(992, 3)
    elif x == 3:
        return np.random.normal(1005, 3)
    elif x == 4:
        return np.random.normal(1003, 3)
    elif x == 5:
        return np.random.normal(999, 3)
    elif x == 6:
        return np.random.normal(998, 3)
    elif x == 7:
        return np.random.normal(1008, 3)
    
synthetic_data["attr5"] = synthetic_data["clus_group_gt"].apply(lambda x: get_attr5(x))
o = open("sticc-synthetic-data.txt", "w")

for i, r in synthetic_data.iterrows():
    o.write(str(r["attr1"]) + "," + str(r["attr2"]) + "," + str(r["attr3"]) + "," + str(r["attr4"]) + "," + str(r["attr5"]) + "\n")
    
o.close()

o = open("sticc-synthetic-labels.txt", "w")

for i, r in synthetic_data.iterrows():
    o.write(str(r["clus_group"]) + "\n")
    
o.close()

pts_all = []
for pt in synthetic_data.iterrows():
    pts_all.append((pt[1].x, pt[1].y))
kd = libpysal.cg.KDTree(np.array(pts_all))
wnn = libpysal.weights.KNN(kd, 3)

nearest_pt = pd.DataFrame().from_dict(wnn.neighbors, orient="index")
for i in range(nearest_pt.shape[1]):
    nearest_pt = nearest_pt.rename({i:f"n_pt_{i}"}, axis=1)
nearest_pt.head(1)

synthetic_data = synthetic_data.join(nearest_pt)
synthetic_data.head(1)

synthetic_data[["attr1", "attr2", 
                "attr3", 
                "attr4", "attr5", 
                "n_pt_0", "n_pt_1", "n_pt_2"
               ]].to_csv(r'synthetic_data.txt', header=None, index=True, sep=',')

synthetic_data_input = pd.read_table(r'synthetic_data.txt', sep=',', names=["id", "attr1", "attr2", 
                "attr3", 
                "attr4", "attr5", 
                "n_pt_0", "n_pt_1", "n_pt_2"
               ])
synthetic_data_input = synthetic_data_input.set_index("id")
synthetic_data_input.head(1)

synthetic_data[["attr1", "attr2", 
                "attr3", "attr4", "attr5"]] = synthetic_data_input[["attr1", "attr2", 
                "attr3", "attr4", "attr5"]]
synthetic_data.head(1)

w_voronoi = weights.Voronoi.from_dataframe(synthetic_data)
