import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_hist(results_path, property):
    flow_type = results_path.split('/')[1].split('_')[0]
    results = pd.read_csv(results_path)
    mean, std = calc_summary(results)
    colors = sns.color_palette()
    legend_items = []
    for i, column in enumerate(results.columns):
        color = colors[i]
        ax = sns.kdeplot(results[column], fill=False, color=color)
        kdeline = ax.lines[i]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        height = np.interp(mean[i], xs, ys)
        ax.vlines(mean[i], 0, height, color=color, ls='dashed')
        ax.fill_between(xs, 0, ys, facecolor=color, alpha=0.2)
        legend_items.append(mpatches.Patch(color=color, alpha=0.8, label=column))
    ax.set(xlabel=property)
    plt.title(flow_type)
    plt.subplots_adjust(bottom=0.1)

    plt.legend(handles=legend_items)
    plt.xlim((-10, 10))
    plt.show()


def calc_summary(results):
    mean = results.mean()
    std = results.std()
    return mean, std


plot_hist("optimization_results/RealNVP_cond_logP.csv", "logP")
