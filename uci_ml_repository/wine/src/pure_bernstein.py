# Trying to compare CBN with no CBN

import openturns as ot
import openturns.viewer as otv
import pyAgrum as gum
import otagrum as otagr

import numpy as np
from pathlib import Path

def dot_quote_adder(dot_string):
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
    for a in arcs:
        new_dot_string += '    ' + a + '\n'
    new_dot_string += tailer
    return new_dot_string

def get_KS_marginals(data):
    print("Marginal KS")
    dimension = data.getDimension()
    KS = ot.KernelSmoothing(ot.Epanechnikov(), False, 0, False)
    marginals = [KS.build(data.getMarginal(i)) for i in range(dimension)]
    return marginals

def write_graph(graph, file_name="output.dot"):
    ''' Util function to write the graph into a DOT file'''
    with open(file_name, 'w') as fo:
        fo.write(dot_quote_adder(graph.toDot()))

def pairs(data, filename):
    ''' Allows to plot the data for each pair of component random variable'''
    print("  Draw pairs")
    print("    Distribution")
    g = ot.Graph()
    pairs_data = ot.Pairs(data)
    pairs_data.setPointStyle('dot')
    g.add(pairs_data)
    view = otv.View(g,(800,800),square_axes=True)
    view.save(filename)
    print("Saving figure in {}".format(filename))
    view.close()
    print("    Copula")
    g = ot.Graph()
    pairs_data = ot.Pairs((data.rank() + 0.5) / data.getSize())
    pairs_data.setPointStyle('dot')
    g.add(pairs_data)
    view = otv.View(g,(800,800),square_axes=True)
    view.save(filename.parent.joinpath(filename.stem+'_copula'+filename.suffix))
    print("Saving figure in {}".format(filename.parent.joinpath(filename.stem+'_copula'+filename.suffix)))
    view.close()

if __name__ == "__main__":
    # Defining directories to put results and figures
    result_path = Path("../results/")
    result_path.mkdir(parents=True, exist_ok=True)

    figure_path = Path("../figures/")
    figure_path.mkdir(parents=True, exist_ok=True)

    # Data file name
    file_name = '../data/winequality-red.csv'

    # Number of data to draw from model
    size_draw = 1000

    # Loading data
    data_ref = ot.Sample.ImportFromTextFile(file_name, ";")
    data_ref = data_ref.getMarginal([0,1,2,8])
    size = data_ref.getSize()     # Size of data
    dim = data_ref.getDimension() # Dimension of data

    size_draw = min(size, size_draw)
    data_draw = data_ref[0:size_draw]  # Number of realizations taken in order to plot figures

    print("Processing reference data")
    f = figure_path.joinpath("test_pairs_ref.pdf")
    pairs(data_draw, f)

    print("Bernstein copula")
    marginals = get_KS_marginals(data_ref)
    KSBC = ot.ComposedDistribution(marginals, ot.BernsteinCopulaFactory().build(data_ref))

    f = figure_path.joinpath("test_pairs_KSBC.pdf")
    pairs(KSBC.getSample(size_draw), f)

    print("Constructing CBN model")

    print("\tLearning structure")
    learner = otagr.ContinuousMIIC(data_ref) # Using CPC algorithm
    learner.setAlpha(-10)
    dag = learner.learnDAG() # Learning DAG
    write_graph(dag, result_path.joinpath("test_dag.dot"))

    print("\tLearning parameters")
    cbn = otagr.ContinuousBayesianNetworkFactory(ot.KernelSmoothing(ot.Epanechnikov()),
                                                 ot.BernsteinCopulaFactory(),
                                                 dag,
                                                 0.05,
                                                 4,
                                                 False).build(data_ref)

    sample_draw = cbn.getSample(size_draw)  # Sampling data from CBN model

    f = figure_path.joinpath("test_pairs_KSPC.pdf")
    pairs(cbn.getSample(size_draw), f)

    # marginal = cbn.getMarginal([1, 2])
    # conditional = [[r.computeConditionalPDF(x, [y]) for x in np.linspace(-10, 10, 100)] for y in np.linspace(-10, 10, 100)]
    # print(conditional)
