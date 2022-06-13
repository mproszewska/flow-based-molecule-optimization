import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse


def plot_hist(results_path, similarity):
    attr = results_path.split("/")[-1].split("_")[1:5]
    if attr[0] not in ["aromatic", "contains"]:
        attr = attr[0]
    elif "or" in attr:
        attr = "_".join(attr)
    else:
        attr = "_".join(attr[:2])
    print(f"Attr {attr}")
    flow_type = results_path.split("/")[1].split("_")[0]
    results = pd.read_csv(results_path)
    results = results.select_dtypes(["number"])
    cols = [
        col
        for col in results.columns
        if (similarity ^ (not col.startswith("similarity"))) or col == "original"
    ]

    results = results[cols]

    mean, std = calc_summary(results)
    colors = sns.color_palette()
    legend_items = []
    plt.rcParams['font.size'] = '20'
    

    cols = [col for col in cols if col != "original"]
    if attr in ["aromatic_rings", "scaffold"] and not similarity:
        attr_list, col_list = list(), list()
        for column in cols:
            attr_list += results[column].tolist()
            col_list += [column for _ in results[column]]
        df = pd.DataFrame({attr: attr_list, "value": col_list})
        

        ax = sns.histplot(
            df,
            x=attr,
            hue="value",
            discrete=True,
            stat="probability",
            common_norm=False,
            multiple="dodge",
            shrink=0.8,
        )
        sns.move_legend(ax, "upper left")
        ax.legend_.set_title(None)

    else:
        cols = [col for col in cols if col != "original"]
        for i, column in enumerate(cols):
            value = column.replace("similarity_", "")
            if value != "original" and "original" in results:
                idx = 1 - np.isclose(results["original"], float(value), atol=0.2)
            else:
                idx = np.ones(len(results[value]))
            color = colors[i]
            ax = sns.kdeplot(results[idx.astype(bool)][column], fill=False, color=color) 
            kdeline = ax.lines[i]
            mean, std = calc_summary(results[idx.astype(bool)][column])
            xs = kdeline.get_xdata()
            ys = kdeline.get_ydata()
            height = np.interp(mean, xs, ys)
            ax.vlines(mean, 0, height, color=color, ls="dashed")
            ax.fill_between(xs, 0, ys, facecolor=color, alpha=0.2)
            legend_items.append(
                mpatches.Patch(
                    color=color,
                    alpha=0.8,
                    label=f"{value}" if len(value) == 3 else str(value)
                )
            )

            i
        ax.legend(handles=legend_items, loc="upper left")
    ax.set(xlabel="Similarity" if similarity else attr)
    # plt.title(flow_type)
    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()
    if attr == "logP":
        plt.xlim((-10, 15))
    elif attr == "SAS":
        plt.xlim((-1, 10))
    elif attr == "qed":
        plt.xlim((0, 1.1))
    elif attr == "aromatic_rings":
        plt.xlim((0, 6))
    else:
        pass
        #raise ValueError
    if similarity:
        plt.xlim((-0.1, 1.1))
    output_path = results_path.replace(".csv", ".jpg")
    if similarity:
        output_path = output_path.replace(".jpg", "_similarity.jpg")
    plt.savefig(output_path)
    print(f"Saved to {output_path}")
    plt.close()


def calc_summary(results):
    mean = results.mean()
    std = results.std()
    return mean, std


parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, required=True)
parser.add_argument("--generate", action="store_true")
args = parser.parse_args()

plot_hist(args.results_path, False)
if not args.generate:
    plot_hist(args.results_path, True)
