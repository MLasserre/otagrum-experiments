# Script that discretize data in order to compare with otagrum

import pyAgrum as gum
import pyAgrum.skbn as skbn
import sklearn.preprocessing as pp

import numpy as np
from pathlib import Path

def write_graph(graph, file_name="output.dot"):
    ''' Util function to write the graph into a DOT file'''
    with open(file_name, 'w') as fo:
        fo.write(graph.toDot())

if __name__ == "__main__":

    # Defining directories to put results and figures
    # result_path = Path("./results/")
    # result_path.mkdir(parents=True, exist_ok=True)

    # figure_path = Path("./figures/")
    # figure_path.mkdir(parents=True, exist_ok=True)

    # Data file name
    file_name = '../data/samples/gaussian/alarm/r08/sample01.csv'
    clf = skbn.BNClassifier(learningMethod='MIIC', discretizationStrategy='quantile',
                            discretizationNbBins=5, discretizationThreshold=25)

    clf.fit(filename=file_name, targetName="PAP")
    bn = clf.bn
    write_graph(bn)
    # dsc = pp.KBinsDiscretizer(11, strategy='kmeans')

    # Loading data
    # print("Processing reference data")
    # f = figure_path.joinpath("pairs_ref.pdf")
    # pairs(data_draw, f)

    # learner = gum.BNLearner(file_name)
    # learne.usePC()
    # bn = learner.learnBN()
    # print("Constructing CBN model")

    # sample_draw = cbn.getSample(size_draw)  # Sampling data from CBN model

    # f = figure_path.joinpath("pairs_KSPC.pdf")
    # pairs(cbn.getSample(size_draw), f)
