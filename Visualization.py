import numpy as np
import pandas as pd
import os
import scipy.stats as st
from sklearn.cluster import KMeans

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

from matplotlib import pyplot as plt

from collections import Counter

import copy

from scipy.spatial import Voronoi, voronoi_plot_2d

def get_sticc_result(ground_truth, filename, title='STICC'):
    pred_labels = pd.read_table(filename, names=["group"])
    return adjusted_rand_score(pred_labels.group, ground_truth[:,2]), normalized_mutual_info_score(pred_labels.group, ground_truth[:,2])

def batch_experiment_clean_kmeans(rate):
    for i in range(10):
        data = np.load("defective_data/synthetic-data-r-{}-{:d}.npz".format(rate, i))
        ground_truth = data["gt"]
        features = data["clean"]
        kmeans = KMeans(n_clusters=7, n_init=300).fit(features)
        ari, nmi = adjusted_rand_score(kmeans.labels_, ground_truth[:,2]), normalized_mutual_info_score(kmeans.labels_, ground_truth[:,2])
        print(ari, nmi)
        markersize = 10
        plt.axis("off")
        plt.scatter(ground_truth[:,0], ground_truth[:,1], label=kmeans.labels_, c=kmeans.labels_, cmap="Set1", s=markersize)
        plt.show()

def compute_error(true_labels, pred_labels):
    tmp = np.ones_like(true_labels)
    for i in range(1,8):
        #print(i)
        idx = np.where(true_labels==i)
        pred_lab = pred_labels[idx].flatten().tolist()
        tid = Counter(pred_lab).most_common()[0][0]
        tidx = (true_labels==i) & (pred_labels==tid)
        #print(tidx)
        #print(len(np.where(tidx == True)[0]))
        tmp[tidx] = 0
    return tmp

def ground_truth_info(ground_truth):
    print(ground_truth)
    print(ground_truth.shape)
    gt_locs = ground_truth[:,:2]
    vor = Voronoi(gt_locs)
    voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)

def draw_results(tmp, sizes):
    #pred_labels = pd.read_table(filename, names=["group"]).values.flatten()
    bsize = 1
    fsize = 50
    
    balpha = 0.5
    falpha = 0.5
    
    plt.figure()
    plt.axis("off")

    cdict = {0: 'grey', 1: 'red'}
    ldict = {0: 'correct', 1: 'wrong'}
    adict = {0: 0.3, 1: 1.0}

    print(tmp)

    # i = 0
    # idx = np.where(tmp == 0)
    # plt.scatter(ground_truth[idx,0], ground_truth[idx,1], label=ldict[i], 
    #                 # c=cdict[i], alpha=balpha + falpha*sizes[idx], s=bsize + fsize*sizes[idx])
    #                 c="grey", s=1)

    # i = 1
    # idx = np.where(tmp == 1)
    # plt.scatter(ground_truth[idx,0], ground_truth[idx,1], label=ldict[i], 
    #                 # c=cdict[i], alpha=balpha + falpha*sizes[idx], s=bsize + fsize*sizes[idx])
    #                 c=i * (falpha*sizes[idx]), 
    #             s=i*(bsize + fsize*sizes[idx]), 
    #             cmap="RdYlGn_r")

    plt.tricontour(ground_truth[:,0], ground_truth[:,1], sizes)

    #plt.legend()
    plt.colorbar()
    plt.show()



data = np.load("defective_data/synthetic-data-r-0.05-0.npz")
ground_truth = data["gt"]

batch_experiment_clean_kmeans("0.05")

ground_truth[:,1].shape

tmp = np.zeros_like(ground_truth[:,2])
for i in range(10):
    data = np.load("defective_data/synthetic-data-r-{}-{:d}.npz".format("0.30", i))
    ground_truth = data["gt"]
    features = data["ood"]
    kmeans = KMeans(n_clusters=7, n_init=300).fit(features)
    tmp += compute_error(ground_truth[:,2], kmeans.labels_)

tmp /= 10

print(np.unique(tmp, return_counts=True))

sizes = copy.deepcopy(tmp)
tmp[tmp>0] = 1

draw_results(tmp, sizes)


rate = "0.30"
tmp = np.zeros_like(ground_truth[:,2])
for i in range(10):
    filename = "results/result-synthetic-data-ood-glasso-{}-{}.txt".format(rate, i)
    pred_labels = pd.read_table(filename, names=["group"]).values.flatten()
    tmp += compute_error(ground_truth[:,2], pred_labels)

tmp /= 10

print(tmp)

print(np.unique(tmp, return_counts=True))

sizes = copy.deepcopy(tmp)
tmp[tmp>0] = 1

draw_results(tmp, sizes)


interp, rate = "localmean", "0.30"
tmp = np.zeros_like(ground_truth[:,2])
for i in range(10):
    filename = "results/result-synthetic-data-ood-stable-static-{}-{}-{}.txt".format(interp, rate, i)
    pred_labels = pd.read_table(filename, names=["group"]).values.flatten()
    tmp += compute_error(ground_truth[:,2], pred_labels)

tmp /= 10

print(np.unique(tmp, return_counts=True))

sizes = copy.deepcopy(tmp)
tmp[tmp>0] = 1

draw_results(tmp, sizes)

interp, rate = "localmean", "0.30"
tmp = np.zeros_like(ground_truth[:,2])
for i in range(10):
    filename = "results/result-synthetic-data-ood-stable-dynamic-{}-{}-{}.txt".format(interp, rate, i)
    pred_labels = pd.read_table(filename, names=["group"]).values.flatten()
    tmp += compute_error(ground_truth[:,2], pred_labels)

tmp /= 10

print(np.unique(tmp, return_counts=True))

sizes = copy.deepcopy(tmp)
tmp[tmp>0] = 1

draw_results(tmp, sizes)


rate, mask_rate = "0.30", "0.10"
tmp = np.zeros_like(ground_truth[:,2])
for i in range(10):
    filename = "results/result-synthetic-data-missing-stable-dynamic-{}-{}.txt".format(rate, i)
    pred_labels = pd.read_table(filename, names=["group"]).values.flatten()
    tmp += compute_error(ground_truth[:,2], pred_labels)

tmp /= 10

print(np.unique(tmp, return_counts=True))

sizes = copy.deepcopy(tmp)
tmp[tmp>0] = 1

draw_results(tmp, sizes)

np.array((8+10, 0, 20+22, 17, 0, 0, 0)) / 24




data = np.loadtxt("defective_data/synthetic-data-clean-0.30-1.txt", delimiter=",")
fs, locs = data[:,:5], data[:,-2:]
plt.axis("off")
plt.scatter(locs[:,0], locs[:,1], c=fs[:,0], cmap="RdYlGn", s=5)
plt.colorbar
plt.savefig("figures/feature-clean.png")

data = np.loadtxt("defective_data/synthetic-data-extreme-0.30-1.txt", delimiter=",")
data.shape
fs, locs = data[:,:5], data[:,-2:]
plt.axis("off")
plt.scatter(locs[:,0], locs[:,1], c=fs[:,0], cmap="RdYlGn", s=5)
plt.colorbar
plt.savefig("figures/feature-info.png")

plt.axis("off")
plt.scatter(locs[:,0], locs[:,1], c=ground_truth[:,-1], s=5, cmap="Set1")
plt.savefig("figures/spatial-info.png")

labels = ground_truth[:,-1]
np.unique(ground_truth[:,-1])
means = np.zeros_like(labels)
for i in range(1,8):
    means[labels == i] = np.mean(fs[labels == i,0])
means
diff = np.abs(fs[:,0] - means)
plt.hist(diff, bins=20)

detect = np.zeros_like(diff)
detect[diff > 2.5] = 1
plt.axis("off")
plt.scatter(locs[detect==0,0], locs[detect==0,1], c="bisque", s=5)
plt.scatter(locs[detect==1,0], locs[detect==1,1], c="red", s=5)
plt.savefig("figures/detect-phase-1.png")

plt.axis("off")
plt.scatter(locs[detect==0,0], locs[detect==0,1], c="bisque", s=5)
plt.scatter(locs[detect==1,0], locs[detect==1,1], c="red", s=5)
plt.savefig("figures/detect-phase-2.png")