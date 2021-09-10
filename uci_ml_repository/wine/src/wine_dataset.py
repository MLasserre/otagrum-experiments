#!/usr/bin/env python
# coding: utf-8

import openturns as ot
import openturns.viewer as otv
import pyAgrum as gum
import otagrum as otagr

from pathlib import Path
import numpy as np

import graphviz
import pydotplus
import matplotlib.pyplot as plt


# DEFINING UTIL FUNCTIONS

def write_graph(graph, file_name="output.dot"):
    ''' Util function to write the graph into a DOT file'''
    with open(file_name, 'w') as fo:
        fo.write(dot_quote_adder(graph.toDot()))


def dot_quote_adder(dot_string):
    # Need it because toDot() method doesn't put quotes around variable names (yet!)
    new_dot_string = ''
    dot_string = dot_string.splitlines()
    header = dot_string[0]
    body = dot_string[1:-1]
    tailer = dot_string[-1]

    arcs = []
    for b in body:
        nodes = b.replace('    ', '').split('->')
        if len(nodes) == 2:
            arcs.append(nodes)

    arcs = ['\"' + a[0] + '\"->\"' + a[1] + '\"' for a in arcs]
    new_dot_string += header + '\n'
    new_dot_string += "    node [style=filled, fillcolor=\"#E9E9E9\", penwidth=1.875, "
    new_dot_string += "fontsize=14, fontname=\"times-bold\"]" + '\n'
    new_dot_string += "    edge [penwidth=1.5]" + '\n'
    for a in arcs:
        new_dot_string += '    ' + a + '\n'
    new_dot_string += tailer
    
    return new_dot_string


def pairs(data, filename):
    ''' Allows to plot the data for each pair of component random variable'''
    print("  Draw pairs")
    print("    Distribution")
    pairs_data = ot.VisualTest.DrawPairs(data)
    otv.View(pairs_data).save(filename)
    print("Saving figure in {}".format(filename))
    print("    Copula")
    pairs_data = ot.VisualTest.DrawPairs((data.rank() + 0.5) / data.getSize())
    print(filename)
    print(filename.stem+'_copula'+filename.suffix)
    print(filename.parent.joinpath(filename.stem+'_copula'+filename.suffix))
    otv.View(pairs_data).save(filename.parent.joinpath(filename.stem+'_copula'+filename.suffix))
    print("Saving figure in {}".format(filename.parent.joinpath(filename.stem+'_copula'+filename.suffix)))


# SETTING DATA, RESULTS AND FIGURE PATHS 

location = Path('..') 

data_path = location.joinpath("data/")

result_path = location.joinpath("results/")
result_path.mkdir(parents=True, exist_ok=True)

structure_path = result_path.joinpath("structures/")
structure_path.mkdir(parents=True, exist_ok=True)

figure_path = location.joinpath("figures/")
figure_path.mkdir(parents=True, exist_ok=True)


# LOADING DATA

# Data file name
file_name = 'winequality-red.csv'

# Loading data
data_ref = ot.Sample.ImportFromTextFile(str(data_path.joinpath(file_name)), ";")
size = data_ref.getSize()     # Size of data
dim = data_ref.getDimension() # Dimension of data


# REFERENCE DATA

# Number of data to draw from model
size_draw = 1000

size_draw = min(size, size_draw)
data_draw = data_ref[0:size_draw]  # Number of realizations taken in order to plot figures

print("Processing reference data")
f = figure_path.joinpath("pairs_ref.pdf")
pairs(data_draw, f)


#####################LEARNING CBN MODEL############################

# LEARNING STRUCTURE #

# CBIC Algorithm
learner = otagr.TabuList(data_ref, 2, 10, 2) # Using CPC algorithm
cbic_dag = learner.learnDAG() # Learning DAG
write_graph(cbic_dag, structure_path.joinpath("cbic_dag.dot"))


# CPC Algorithm
learner = otagr.ContinuousPC(data_ref, 4, 0.05) # Using CPC algorithm
cpc_dag = learner.learnDAG() # Learning DAG
write_graph(cpc_dag, structure_path.joinpath("cpc_dag.dot"))


# CMIIC ALGORITHM

# Alpha = 0.01
learner = otagr.ContinuousMIIC(data_ref) # Using CPC algorithm
learner.setAlpha(0.01)
cmiic_dag = learner.learnDAG() # Learning DAG
write_graph(cmiic_dag, structure_path.joinpath("cmiic_dag_1.dot"))

# Alpha = 0.02
learner = otagr.ContinuousMIIC(data_ref) # Using CPC algorithm
learner.setAlpha(0.02)
cmiic_dag = learner.learnDAG() # Learning DAG
write_graph(cmiic_dag, structure_path.joinpath("cmiic_dag_2.dot"))

# Alpha = 0.03
learner = otagr.ContinuousMIIC(data_ref) # Using CPC algorithm
learner.setAlpha(0.03)
cmiic_dag = learner.learnDAG() # Learning DAG
write_graph(cmiic_dag, structure_path.joinpath("cmiic_dag_3.dot"))

# Alpha = 0.04
learner = otagr.ContinuousMIIC(data_ref) # Using CPC algorithm
learner.setAlpha(0.04)
cmiic_dag = learner.learnDAG() # Learning DAG
write_graph(cmiic_dag, structure_path.joinpath("cmiic_dag_4.dot"))

# Alpha = 0.05
learner = otagr.ContinuousMIIC(data_ref) # Using CPC algorithm
learner.setAlpha(0.05)
cmiic_dag = learner.learnDAG() # Learning DAG
write_graph(cmiic_dag, structure_path.joinpath("cmiic_dag_5.dot"))

# Alpha = 0.06
learner = otagr.ContinuousMIIC(data_ref) # Using CPC algorithm
learner.setAlpha(0.06)
cmiic_dag = learner.learnDAG() # Learning DAG
write_graph(cmiic_dag, structure_path.joinpath("cmiic_dag_6.dot"))


'''
#################LEARNING PARAMETERS########################

#REF DATA
pairs(data_draw, f)

# CBIC
cbic_cbn = otagr.ContinuousBayesianNetworkFactory(ot.KernelSmoothing(ot.Epanechnikov()),
                                             ot.BernsteinCopulaFactory(),
                                             cbic_dag,
                                             0.05,
                                             4,
                                             False).build(data_ref)
f = figure_path.joinpath("pairs_KSBIC.pdf")
pairs(cbic_cbn.getSample(size_draw), f)

# CPC
cpc_cbn = otagr.ContinuousBayesianNetworkFactory(ot.KernelSmoothing(ot.Epanechnikov()),
                                             ot.BernsteinCopulaFactory(),
                                             cpc_dag,
                                             0.05,
                                             4,
                                             False).build(data_ref)
f = figure_path.joinpath("pairs_KSPC.pdf")
pairs(cpc_cbn.getSample(size_draw), f)

# CMIIC
cmiic_cbn = otagr.ContinuousBayesianNetworkFactory(ot.KernelSmoothing(ot.Epanechnikov()),
                                             ot.BernsteinCopulaFactory(),
                                             cmiic_dag,
                                             0.05,
                                             4,
                                             False).build(data_ref)
f = figure_path.joinpath("pairs_KSMIIC.pdf")
pairs(cmiic_cbn.getSample(size_draw), f)
'''
