import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse


def plot_hist(results_path):
    property = results_path.split("_")[-4]
    flow_type = results_path.split("/")[1].split("_")[0]
    results = pd.read_csv(results_path).select_dtypes(['number'])

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
        ax.vlines(mean[i], 0, height, color=color, ls="dashed")
        ax.fill_between(xs, 0, ys, facecolor=color, alpha=0.2)
        legend_items.append(
            mpatches.Patch(
                color=color, alpha=0.8, label=f"{column} (mean={mean[i]:.2f})"
            )
        )
    ax.set(xlabel=property)
    plt.title(flow_type)
    plt.subplots_adjust(bottom=0.1)

    plt.legend(handles=legend_items)
    if property == "logP": plt.xlim((-10, 10))
    elif property == "SAS": plt.xlim((-1, 10))
    elif property == "qed": plt.xlim((0, 1.1))
    else: raise ValueError
    output_path = results_path.replace(".csv", ".jpg")
    plt.savefig(output_path)


def calc_summary(results):
    mean = results.mean()
    std = results.std()
    return mean, std


parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, required=True)

args = parser.parse_args()

plot_hist(args.results_path)
