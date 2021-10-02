from pipeline import Pipeline
import otagrum as otagr
from otagrum import CorrectedMutualInformation as cmi
import numpy as np
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')
mpl.rcParams.update({'font.size': 22,
                     # 'xtick.major.width': 3.2,
                     # 'xtick.major.size': 14,
                     # 'ytick.major.width': 3.2,
                     # 'ytick.major.size': 14,
                     # 'axes.linewidth': 2,
                     # 'figure.autolayout': True,
                     # 'legend.frameon': False,
                     # 'legend.borderpad': 0.001,
                     # 'legend.borderaxespad': 0.2,
                     # 'legend.handlelength': 1.1,
                     'legend.labelspacing': 0.2
                    })

def get_domain(left, right):
    if left%10!=0 or right%10!=0:
        print("left and right must be multiples of 10!")
    if left > right:
        print("left must be less than right!")

    a = 0
    temp = left
    while (temp%10) == 0:
        temp = temp//10
        a = a + 1

    b = 0
    temp = right
    while (temp%10) == 0:
        temp = temp/10
        b = b + 1
         
    x = left
    domain = [x]
    while x < right:
        while x < 10**(a+1) and x < right:
            x = x + 10**a
            domain.append(x)
        a = a + 1
    return domain

distributions = ['gaussian', 'student', 'dirichlet']
structures = ['alarm']

size_min = {'alarm':100}
size_max = {'alarm':15000}
n_points = {'alarm': 15}
n_restart = {'alarm':5}
xlim = {'alarm':6000}
ylim = {'alarm':80}

correlations = np.round(np.linspace(0.8, 0.8, 1), decimals=1)

bin_range = np.arange(5, 21, 5, dtype=int)
pls = {}
for b in bin_range:
    pls[b] = Pipeline('dmiic', dis_method='quantile', nbins=b, threshold=25)

plot_style = {'linewidth':2.}

for structure in structures:
    print('Structure :', structure)
    for key in pls:
        pls[key].setDataStructure(structure)
        pls[key].setResultDomain(size_min[structure],
                                 size_max[structure],
                                 n_points[structure],
                                 n_restart[structure])

    for distribution in distributions:
        print('Distribution :', distribution)
        if distribution == 'gaussian' or distribution == 'student':
            for correlation in correlations:
                print('Correlation :', correlation)
                apath = os.path.join('../figures/',
                                     distribution,
                                     structure,
                                     'r'+str(correlation).replace('.', ''))
                Path(apath).mkdir(parents=True, exist_ok=True)
                for key in pls:
                    pls[key].setDataDistribution(distribution, r=correlation)
                    pls[key].generate_data()
                    print('dmiic with {} bins'.format(key), flush=True)
                    pls[key].computeStructuralScore('skelF')
                    pls[key].computeStructuralScore('hamming')

                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], xlim[structure]])
                ax.set_ylim(0,1)
                for key in pls:
                    pls[key].plotScore('skelF', fig, ax, **plot_style, label=str(key))
                handles, labels = ax.get_legend_handles_labels()
                handles = [h[0] for h in handles]
                ax.legend(handles, labels, loc='lower right')
                plt.savefig(os.path.join(apath,
                                         '_'.join(['fscore', distribution , structure]) +'discretization_study.pdf'),
                            transparent=True)
                
                
                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], size_max[structure]])
                ax.set_ylim(0,ylim[structure])
                for key in pls:
                    pls[key].plotScore('hamming', fig, ax, **plot_style, label=str(key))
                handles, labels = ax.get_legend_handles_labels()
                handles = [h[0] for h in handles]
                ax.legend(handles, labels, loc='upper right')
                plt.savefig(os.path.join(apath,
                                         '_'.join(['hamming', distribution , structure]) +'.pdf'),
                        transparent=True)

                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], size_max[structure]])
                # ax.set_ylim(0,ylim[structure])
                for key in pls:
                    pls[key].plotTime(fig, ax, **plot_style, label=str(key))
                handles, labels = ax.get_legend_handles_labels()
                handles = [h[0] for h in handles]
                ax.legend(handles, labels)
                plt.savefig(os.path.join(apath,
                                         '_'.join(['time_complexity', distribution , structure]) +'.pdf'),
                        transparent=True)
                
                
        elif distribution == 'dirichlet':
            bpath = os.path.join('../figures/', distribution, structure)
            Path(bpath).mkdir(parents=True, exist_ok=True)
            for key in pls:
                pls[key].setDataDistribution(distribution)
                pls[key].generate_data()
            
                print('dmiic with {} bins'.format(key))
                pls[key].computeStructuralScore('skelF')
                pls[key].computeStructuralScore('hamming')
            
            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], xlim[structure]])
            ax.set_ylim(0,1)
            for key in pls:
                pls[key].plotScore('skelF', fig, ax, **plot_style, label=str(key))
            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] for h in handles]
            ax.legend(handles, labels, loc='lower right')
            plt.savefig(os.path.join(bpath,
                                     '_'.join(['fscore', distribution , structure]) +'.pdf'),
                        transparent=True)
            
            
            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], size_max[structure]])
            ax.set_ylim(0,ylim[structure])
            for key in pls:
                pls[key].plotScore('hamming', fig, ax, **plot_style, label=str(key))
            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] for h in handles]
            ax.legend(handles, labels, loc='upper right')
            print("Saving figure in {}".format(os.path.join(bpath,
                                                            '_'.join(['hamming',
                                                                      distribution,
                                                                      structure,]) + '.pdf')))
            plt.savefig(os.path.join(bpath,
                                     '_'.join(['hamming', distribution , structure]) +'.pdf'),
                        transparent=True)

            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], size_max[structure]])
            # ax.set_ylim(0,yim[structure])
            for key in pls:
                pls[key].plotTime(fig, ax, **plot_style, label=str(key))

            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] for h in handles]
            ax.legend(handles, labels)
            plt.savefig(os.path.join(bpath,
                                     '_'.join(['time_complexity', distribution , structure]) +'.pdf'),
                        transparent=True)
