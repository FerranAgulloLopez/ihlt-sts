import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def show_scatter_plot(labels, results, title, output_path):
    plt.figure()
    plt.scatter(labels, results)
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.xlabel('Gold standard')
    plt.ylabel('Similarity')
    plt.title(title)
    if output_path is not None:
        plt.savefig(output_path + '/' + title)
    plt.show()


def show_correlation_plot(config, labels, metrics_results, title, output_path):
    plt.figure(figsize=(40,40))
    colnames = [x['name'] for x in config] + ['gs']
    labels = np.expand_dims(np.asarray(labels), axis=0).T
    values = np.concatenate((metrics_results, labels), axis=1)
    values = pd.DataFrame(data=values, index=list(range(0,values.shape[0])), columns=colnames)
    corr = values.corr()
    sns.heatmap(corr, annot=True,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.xlabel('Gold standard')
    plt.ylabel('Similarity')
    plt.title(title)
    if output_path is not None:
        plt.savefig(output_path + '/' + title)
    plt.show()