# Script that discretize data in order to compare with otagrum

import pyAgrum as gum
import openturns as ot
import otagrum as otagr
import pyAgrum.skbn as skbn

import numpy as np
# import pandas as pd
import tempfile as tf
import time
import os

from pathlib import Path

def write_graph(graph, file_name="output.dot"):
    ''' Util function to write the graph into a DOT file'''
    with open(file_name, 'w') as fo:
        fo.write(graph.toDot())

def learnDAG(sample, dis_method='quantile', nbins=5, threshold=25):
    # data = pd.read_csv(file_name, nrows=size)

    names = list(sample.getDescription())

    csvfile = tf.NamedTemporaryFile(delete=False)
    csvfilename = csvfile.name + '.csv'
    csvfile.close()

    sample.exportToCSVFile(csvfilename, ',')

    start = time.time()
    discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod=dis_method,
                                    defaultNumberOfBins=nbins,
                                    discretizationThreshold=threshold)

    variables = [discretizer.createVariable(name, sample.getMarginal([name])) for name in names]

    bn = gum.BayesNet()
    for variable in variables:
        bn.add(variable)

    learner = gum.BNLearner(csvfilename, bn)
    learner.useMIIC()
    learner.useNMLCorrection()

    dag = learner.learnDAG()
    ndag = otagr.NamedDAG(dag, names)

    end = time.time()

    os.remove(csvfilename)

    return ndag, start, end

if __name__ == '__main__':
    data = ot.Sample.ImportFromTextFile('../data/samples/gaussian/alarm/r08/sample01.csv', ',')
    data = (data.rank()+1)/(data.getSize()+2)
    sample = data[0:1000]
    dag, s, e = learnDAG(sample)
    write_graph(dag)
