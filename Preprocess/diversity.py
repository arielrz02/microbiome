import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skbio.diversity import beta_diversity
from skbio.diversity import alpha_diversity

from Plot.plot_3D import PCoA_and_plot
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from LearningMethods.textreeCreate import create_tax_tree


class Diversity(object):

    def __init__(self, OTUfile):
        mapping_file = CreateOtuAndMappingFiles(OTUfile, tags_file_path=None)
        otu_df = mapping_file.otu_features_df.copy()
        self.sample_ids = otu_df.index[:-1]
        self.otu_ids = otu_df.iloc[[-1]]
        self.otu_df = otu_df.iloc[:-1]


    def compute_beta(self, metric="unweighted_unifrac"):
        if "unifrac" not in metric:
            dist_mat = beta_diversity(metric, self.otu_df, self.sample_ids)
        else:
            dist_mat = self.__beta_unifrac(metric)
        return dist_mat


    def __beta_unifrac(self,  method):
        unweighted = (method == "unweighted_unifrac")
        distmat = np.zeros(shape=(len(self.otu_df), len(self.otu_df)))
        trees = []
        self.otu_df.columns = self.otu_ids.iloc[0].T
        for sample in self.otu_df.index:
            series = self.otu_df.loc[sample]
            trees.append(create_tax_tree(series))
        trees_partial = trees.copy()

        for ida, tree_a in enumerate(trees):
            trees_partial.remove(tree_a)

            for idb, tree_b in enumerate(trees_partial):
                shared_count = 0
                total = 0

                for node_a, node_b in zip(tree_a.nodes, tree_b.nodes):
                    if unweighted:
                        add = 1
                    else:
                        add = node_a[1] + node_b[1]

                    total += add

                    if node_a[1] * node_b[1] == 0 and node_a[1] + node_b[1] != 0:
                        shared_count += add

                distmat[ida][idb+ida+1] = shared_count/total
                distmat[idb+ida+1][ida] = shared_count/total
        distdf = pd.DataFrame(distmat)
        distdf.index, distdf.columns = self.otu_df.index, self.otu_df.index
        return distdf


    def compute_alpha(self, metric="shannon"):
        if metric == 'shannon':
            otu_df_alpha = self.otu_df.replace(0, 1)
        else:
            otu_df_alpha = self.otu_df
        dist_series = alpha_diversity(metric, otu_df_alpha, self.sample_ids)
        return dist_series

    def plot_beta(self, metric="unweighted_unifrac", title="beta_diversity", folder="Plot", **kwargs):
        df = self.compute_beta(metric)
        PCoA_and_plot(df, title=title, folder=folder, **kwargs)


if __name__ == "__main__":
    diversity = Diversity("OTU.csv")
    diversity.compute_alpha()
    diversity.plot_beta(metric="unweighted_unifrac")