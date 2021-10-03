import openturns as ot
from otagrum import NamedDAG

from pathlib import Path
import os
import tempfile as tf
import time

from graph_utils import read_graph

def learn_lgbn(sample, restart=20, maxp=5):

    names = list(sample.getDescription())
    
    csvfile = tf.NamedTemporaryFile(delete=False)
    csvfilename = csvfile.name + '.csv'
    csvfile.close()

    dotfile = tf.NamedTemporaryFile(delete=False)
    dotfilename = dotfile.name + '.dot'
    dotfile.close()

    sample.exportToCSVFile(csvfilename, ',')

    start = time.time()
    os.system("Rscript learn_LGBN.R " + str(csvfilename) + ' '
                                         + str(dotfilename) + ' '
                                         + str(restart) + ' '
                                         + str(maxp))
    end = time.time()

    dag, names = read_graph(dotfilename)

    os.remove(csvfilename)
    os.remove(dotfilename)

    return NamedDAG(dag, names), start, end

if __name__ == '__main__':
    data = ot.Sample.ImportFromTextFile('../data/samples/gaussian/alarm/r08/sample01.csv', ',')
    data = (data.rank()+1)/(data.getSize()+2)
    sample = data[0:1000]
    ndag, s, e = learn_lgbn(sample)
    print(ndag)
