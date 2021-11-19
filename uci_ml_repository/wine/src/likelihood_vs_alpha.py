#!/usr/bin/env python
# coding: utf-8

import openturns as ot
import openturns.viewer as otv
import pyAgrum as gum
import otagrum as otagr

from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

import graphviz
import pydotplus
import matplotlib.pyplot as plt

import sys


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

np.random.seed(42)
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
kf = KFold(n_splits=100)
size = data_ref.getSize()     # Size of data
dim = data_ref.getDimension() # Dimension of data

# data_ref = (data_ref.rank() +1)/(size + 2)

#alphas = np.arange(10, 501, 5)/1000
alphas = np.arange(1e-2, 5e-1, 1e-2)
# alphas = [0.4]

ot.ResourceMap.SetAsUnsignedInteger("ContinuousBayesianNetworkFactory-MaximumDiscreteSupport", 0) 
ot.ResourceMap.SetAsBool("ContinuousBayesianNetworkFactory-UseBetaCopula", sys.argv[1] == "beta")
suffix = sys.argv[1]
print("suffix=", suffix)

likelihood_curves = []
bic_curves = []
splits = kf.split(data_ref)
# for (i, (train, test)) in enumerate(splits):
for (i, (train, test)) in enumerate(list(splits)):
    print("Learning with fold number {}".format(i))
    likelihood_curve = []
    bic_curve = []
    for alpha in alphas:
        ot.Log.Show(ot.Log.NONE)
        print("\tLearning with alpha={}".format(alpha))
        learner = otagr.ContinuousMIIC(data_ref.select(train)) # Using CMIIC algorithm
        learner.setAlpha(alpha)
        cmiic_dag = learner.learnDAG() # Learning DAG
        ot.Log.Show(ot.Log.NONE)
        if True:
            cmiic_cbn = otagr.ContinuousBayesianNetworkFactory(ot.KernelSmoothing(ot.Normal()),
                                                               ot.BernsteinCopulaFactory(),
                                                               cmiic_dag,
                                                               0.05,
                                                               4,
                                                               False).build(data_ref.select(train))
        else:
            cmiic_cbn = otagr.ContinuousBayesianNetworkFactory(ot.NormalFactory(),
                                                               ot.NormalCopulaFactory(),
                                                               cmiic_dag,
                                                               0.05,
                                                               4,
                                                               False).build(data_ref.select(train))
        # sampled = cmiic_cbn.getSample(1000)
        # sampled = (sampled.rank() +1)/(sampled.getSize()+2)
        # pairs(sampled, figure_path.joinpath('pairs_test.pdf')
        ll = 0
        s = 0
        for point in data_ref.select(test):
            point_ll = cmiic_cbn.computeLogPDF(point)
            if np.abs(point_ll) <= 10e20:
                s+=1
                ll += point_ll
            else:
                print("pb point=", point, "log pdf=", point_ll)
        ll /= s
        n_arc = cmiic_dag.getDAG().sizeArcs()
        # ll = cmiic_cbn.computeLogPDF(data_ref.select(test)).computeMean()[0]
        #bic = ll - 0.5 * np.log(data_ref.select(test).getSize()) * n_arc
        bic = s*ll - 0.5 * np.log(s) * n_arc / s
        print("\t\tLikelihood: {}".format(ll))
        print("\t\tBIC: {}".format(bic))
        likelihood_curve.append(ll)
        bic_curve.append(bic)
    likelihood_curves.append(likelihood_curve)
    bic_curves.append(bic_curve)

print(likelihood_curves)
likelihood_mean_curve = np.array(likelihood_curves).mean(axis=0)
bic_mean_curve = np.array(bic_curves).mean(axis=0)

likelihood_std_curve = np.array(likelihood_curves).std(axis=0)
bic_std_curve = np.array(bic_curves).std(axis=0)

fig, ax = plt.subplots()

# x_major_ticks = np.arange(0, 0.5, 0.05)
# x_minor_ticks = np.arange(0, 0.5, 0.01)

# y_major_ticks = np.arange(0, 25, 5)
# y_minor_ticks = np.arange(0, 25, 1)

# ax.set_xticks(x_major_ticks)
# ax.set_xticks(x_minor_ticks, minor=True)
# ax.set_yticks(y_major_ticks)
# ax.set_yticks(y_minor_ticks, minor=True)

# ax.set_xlim(0, 0.5)
# ax.set_ylim(0, 25)

ax.errorbar(alphas, likelihood_mean_curve, yerr=likelihood_std_curve)
plt.savefig(figure_path.joinpath("likelihood_vs_alpha_" + suffix + ".pdf"), transparent=True)
print("Saving figure in {}".format(figure_path.joinpath("likelihood_vs_alpha_" + suffix + ".pdf")))

fig, ax = plt.subplots()
ax.errorbar(alphas, bic_mean_curve, yerr=bic_std_curve)
plt.savefig(figure_path.joinpath("bic_vs_alpha_" + suffix + ".pdf"), transparent=True)
print("Saving figure in {}".format(figure_path.joinpath("bic_vs_alpha_" + suffix + ".pdf")))
