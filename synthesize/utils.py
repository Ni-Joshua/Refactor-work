import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def create_random_mask(n, d, miss_percent=0.1):
    num_entries = n * d
    mask = np.full(num_entries, True)
    mask[:int(num_entries * miss_percent)] = False
    np.random.shuffle(mask)
    mask = mask.reshape(n, d)
    return mask


def mean_imputation(X):
    # Use mean of each variable to impute the missing entries
    X = np.copy(X)
    X_df = pd.DataFrame(X)
    for col in X_df:
        X_df[col] = X_df[col].fillna(X_df[col].mean(skipna=True))
    return X_df.values


def compute_mse(theta_true, theta_est):
    return np.sum((theta_true - theta_est) **2)


def plot_solution(theta_true, theta_est):
    """
    Modified from: https://github.com/ignavierng/golem/blob/main/src/utils/utils.py
    """
    fig, axes = plt.subplots(figsize=(7, 3), ncols=2)

    # Plot ground truth
    im = axes[0].imshow(theta_true, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[0].set_title("Ground truth", fontsize=13)
    axes[0].tick_params(labelsize=13)

    # Plot estimated solution
    im = axes[1].imshow(theta_est, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[1].set_title("Estimated solution", fontsize=13)
    axes[1].set_yticklabels([])    # Remove yticks
    axes[1].tick_params(labelsize=13)

    # Adjust space between subplots
    fig.subplots_adjust(wspace=0.1)

    # Colorbar (with abit of hard-coding)
    im_ratio = 3 / 10
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.05*im_ratio, pad=0.035)
    cbar.ax.tick_params(labelsize=13)
    # plt.show()