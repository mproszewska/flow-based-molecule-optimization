import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_hist(results_path, property):
    results = pd.read_csv(results_path)
    ax = sns.displot(results, kind="kde", fill=True)
    ax.set(xlabel=property)
    plt.subplots_adjust(bottom=0.1)
    plt.xlim((-10,10))
    plt.show()


plot_hist("optimization_results/logP.csv", "logP")
