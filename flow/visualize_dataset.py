import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse


def plot_hist(path, attr):
    data = pd.read_csv(path)
    data["set"] = np.where(data.index < 240000, "train", "test")
    data["set"] = data["set"].astype("category")
    data = data[[attr, "set"]]
    print(data)
    colors = sns.color_palette()
    legend_items = []
    if attr == "aromatic_rings" or attr.startswith("contains"):
        ax = sns.histplot(
            data,
            x=attr,
            hue="set",
            hue_order=["train", "test"],
            discrete=True,
            stat="probability",
            common_norm=False,
            multiple="dodge",
            shrink=0.8,
        )

    else:
        ax = sns.kdeplot(
            data[attr],
            hue=data["set"],
            hue_order=["train", "test"],
            fill=True,
            legend=True,
            common_norm=False,
        )
    ax.legend_.set_title(None)
    ax.set(xlabel=attr)
    plt.title("Dataset")
    plt.subplots_adjust(bottom=0.1)

    # plt.legend(handles=legend_items)
    if attr == "logP":
        plt.xlim((-10, 10))
    elif attr == "SAS":
        plt.xlim((-1, 10))
    elif attr == "qed":
        plt.xlim((0, 1.1))
    elif attr == "aromatic_rings":
        plt.xlim((-1, 6))
    elif attr.startswith("contains"):
        plt.xlim((-1, 2))
    else:
        raise ValueError
    output_path = path.replace(".csv", f"_{attr}.jpg")
    plt.savefig(output_path)
    print(f"Saved to {output_path}")
    plt.close()


def calc_summary(results):
    mean = results.mean()
    std = results.std()
    return mean, std


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--attr", type=str, required=True)
args = parser.parse_args()


plot_hist(args.path, args.attr)
