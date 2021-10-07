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
size_max = {'alarm':200}
n_points = {'alarm': 5}
n_restart = {'alarm':5}
xlim = {'alarm':6000}
ylim = {'alarm':80}

correlations = np.round(np.linspace(0.8, 0.8, 1), decimals=1)

cmiic_gaussian = Pipeline('cmiic',
                          cmode=cmi.CModeTypes_Gaussian,
                          kmode=cmi.KModeTypes_Naive)

cmiic_bernstein = Pipeline('cmiic',
                           cmode=cmi.CModeTypes_Bernstein,
                           kmode=cmi.KModeTypes_Naive)

cpc = Pipeline('cpc', binNumber=5, alpha=0.05)

cbic_gaussian = Pipeline('cbic', max_parents=4, hc_restart=5,
                         cmode=cmi.CModeTypes_Gaussian)

dmiic = Pipeline('dmiic', dis_method='quantile', nbins=5, threshold=25)
gbn = Pipeline('gbn', maxp=4, restart=5)

plot_style_bernstein = {'linewidth':2.,
                        'linestyle':'-',
                        'color':'royalblue',
                        'label':'b-miic'}

plot_style_gaussian = {'linewidth':2.,
                       'linestyle':'--',
                       'color':'goldenrod',
                       'label':'g-miic'}

plot_style_cpc = {'linewidth':2.,
                  'linestyle':'-.',
                  'color':'maroon',
                  'label':'cpc'}

plot_style_cbic_gaussian = {'linewidth':2.,
                            'linestyle':(0, (1,1)),
                            'color':'olivedrab',
                            'label':'cbic'}

plot_style_dmiic = {'linewidth':2.,
                    'linestyle':'-.',
                    'color':'red',
                    'label':'d-miic'}

plot_style_gbn = {'linewidth':2.,
                    'linestyle':'-.',
                    'color':'green',
                    'label':'gbn'}

for structure in structures:
    print('Structure :', structure)
    cmiic_gaussian.setDataStructure(structure)
    cmiic_bernstein.setDataStructure(structure)
    cpc.setDataStructure(structure)
    cbic_gaussian.setDataStructure(structure)
    dmiic.setDataStructure(structure)
    gbn.setDataStructure(structure)
    
    cmiic_gaussian.setResultDomain(size_min[structure],
                                   size_max[structure],
                                   n_points[structure],
                                   n_restart[structure])
    cmiic_bernstein.setResultDomain(size_min[structure],
                                    size_max[structure],
                                    n_points[structure],
                                    n_restart[structure])
    cpc.setResultDomain(size_min[structure],
                        size_max[structure],
                        n_points[structure],
                        n_restart[structure])

    cbic_gaussian.setResultDomain(size_min[structure],
                                  size_max[structure],
                                  n_points[structure],
                                  n_restart[structure])

    dmiic.setResultDomain(size_min[structure],
                          size_max[structure],
                          n_points[structure],
                          n_restart[structure])

    gbn.setResultDomain(size_min[structure],
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
                cmiic_gaussian.setDataDistribution(distribution, r=correlation)
                cmiic_bernstein.setDataDistribution(distribution, r=correlation)
                cpc.setDataDistribution(distribution, r=correlation)
                cbic_gaussian.setDataDistribution(distribution, r=correlation)
                dmiic.setDataDistribution(distribution, r=correlation)
                gbn.setDataDistribution(distribution, r=correlation)
                
                cmiic_gaussian.generate_data()
                cmiic_bernstein.generate_data()
                cpc.generate_data()
                cbic_gaussian.generate_data()
                dmiic.generate_data()
                gbn.generate_data()
                
                print('cmiic gaussian', flush=True)
                cmiic_gaussian.computeStructuralScore('skelF')
                cmiic_gaussian.computeStructuralScore('hamming')
                cmiic_gaussian.computeMeanTime()
                
                print('cmiic bernstein', flush=True)
                cmiic_bernstein.computeStructuralScore('skelF')
                cmiic_bernstein.computeStructuralScore('hamming')
                cmiic_bernstein.computeMeanTime()
                
                print('cpc', flush=True)
                cpc.computeStructuralScore('skelF')
                cpc.computeStructuralScore('hamming')
                cpc.computeMeanTime()

                print('cbic gaussian', flush=True)
                cbic_gaussian.computeStructuralScore('skelF')
                cbic_gaussian.computeStructuralScore('hamming')
                cbic_gaussian.computeMeanTime()

                print('dmiic', flush=True)
                dmiic.computeStructuralScore('skelF')
                dmiic.computeStructuralScore('hamming')
                dmiic.computeMeanTime()

                print('gbn', flush=True)
                gbn.computeStructuralScore('skelF')
                gbn.computeStructuralScore('hamming')
                gbn.computeMeanTime()

                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], xlim[structure]])
                ax.set_ylim(0,1)

                cmiic_bernstein.plotMetric('skelF', fig, ax, **plot_style_bernstein)
                cmiic_gaussian.plotMetric('skelF', fig, ax, **plot_style_gaussian)
                cpc.plotMetric('skelF', fig, ax, **plot_style_cpc)
                cbic_gaussian.plotMetric('skelF', fig, ax, **plot_style_cbic_gaussian)
                dmiic.plotMetric('skelF', fig, ax, **plot_style_dmiic)
                gbn.plotMetric('skelF', fig, ax, **plot_style_gbn)

                handles, labels = ax.get_legend_handles_labels()
                handles = [h[0] for h in handles]
                ax.legend(handles, labels, loc='lower right')
                plt.savefig(os.path.join(apath,
                                         '_'.join(['fscore', distribution , structure]) +'.pdf'),
                            transparent=True)
                
                
                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], size_max[structure]])
                ax.set_ylim(0,ylim[structure])

                cmiic_bernstein.plotMetric('hamming', fig, ax, **plot_style_bernstein)
                cmiic_gaussian.plotMetric('hamming', fig, ax, **plot_style_gaussian)
                cpc.plotMetric('hamming', fig, ax, **plot_style_cpc)
                cbic_gaussian.plotMetric('hamming', fig, ax, **plot_style_cbic_gaussian)
                dmiic.plotMetric('hamming', fig, ax, **plot_style_dmiic)
                gbn.plotMetric('hamming', fig, ax, **plot_style_gbn)

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

                cmiic_bernstein.plotMetric('time', fig, ax, **plot_style_bernstein)
                cmiic_gaussian.plotMetric('time', fig, ax, **plot_style_gaussian)
                cpc.plotMetric('time', fig, ax, **plot_style_cpc)
                cbic_gaussian.plotMetric('time', fig, ax, **plot_style_cbic_gaussian)
                dmiic.plotMetric('time', fig, ax, **plot_style_dmiic)
                gbn.plotMetric('time', fig, ax, **plot_style_gbn)

                handles, labels = ax.get_legend_handles_labels()
                handles = [h[0] for h in handles]
                ax.legend(handles, labels)
                plt.savefig(os.path.join(apath,
                                         '_'.join(['time_complexity', distribution , structure]) +'.pdf'),
                        transparent=True)
                
                
        elif distribution == 'dirichlet':
            bpath = os.path.join('../figures/', distribution, structure)
            Path(bpath).mkdir(parents=True, exist_ok=True)

            cmiic_gaussian.setDataDistribution(distribution)
            cmiic_bernstein.setDataDistribution(distribution)
            cpc.setDataDistribution(distribution)
            cbic_gaussian.setDataDistribution(distribution)
            dmiic.setDataDistribution(distribution)
            gbn.setDataDistribution(distribution)

            cmiic_gaussian.generate_data()
            cmiic_bernstein.generate_data()
            cpc.generate_data()
            cbic_gaussian.generate_data()
            dmiic.generate_data()
            gbn.generate_data()
            
            print('cmiic gaussian')
            cmiic_gaussian.computeStructuralScore('skelF')
            cmiic_gaussian.computeStructuralScore('hamming')
            cmiic_gaussian.computeMeanTime()
            
            print('cmiic bernstein')
            cmiic_bernstein.computeStructuralScore('skelF')
            cmiic_bernstein.computeStructuralScore('hamming')
            cmiic_bernstein.computeMeanTime()
            
            print('cpc')
            cpc.computeStructuralScore('skelF')
            cpc.computeStructuralScore('hamming')
            cpc.computeMeanTime()

            print('cbic gaussian')
            cbic_gaussian.computeStructuralScore('skelF')
            cbic_gaussian.computeStructuralScore('hamming')
            cbic_gaussian.computeMeanTime()

            print('dmiic')
            dmiic.computeStructuralScore('skelF')
            dmiic.computeStructuralScore('hamming')
            dmiic.computeMeanTime()

            print('gbn')
            gbn.computeStructuralScore('skelF')
            gbn.computeStructuralScore('hamming')
            gbn.computeMeanTime()
            
            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], xlim[structure]])
            ax.set_ylim(0,1)

            cmiic_bernstein.plotMetric('skelF', fig, ax, **plot_style_bernstein)
            cmiic_gaussian.plotMetric('skelF', fig, ax, **plot_style_gaussian)
            cpc.plotMetric('skelF', fig, ax, **plot_style_cpc)
            cbic_gaussian.plotMetric('skelF', fig, ax, **plot_style_cbic_gaussian)
            dmiic.plotMetric('skelF', fig, ax, **plot_style_dmiic)
            gbn.plotMetric('skelF', fig, ax, **plot_style_gbn)

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

            cmiic_bernstein.plotMetric('hamming', fig, ax, **plot_style_bernstein)
            cmiic_gaussian.plotMetric('hamming', fig, ax, **plot_style_gaussian)
            cpc.plotMetric('hamming', fig, ax, **plot_style_cpc)
            cbic_gaussian.plotMetric('hamming', fig, ax, **plot_style_cbic_gaussian)
            dmiic.plotMetric('hamming', fig, ax, **plot_style_dmiic)
            gbn.plotMetric('hamming', fig, ax, **plot_style_gbn)

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

            cmiic_bernstein.plotMetric('time', fig, ax, **plot_style_bernstein)
            cmiic_gaussian.plotMetric('time', fig, ax, **plot_style_gaussian)
            cpc.plotMetric('time', fig, ax, **plot_style_cpc)
            cbic_gaussian.plotMetric('time', fig, ax, **plot_style_cbic_gaussian)
            dmiic.plotMetric('time', fig, ax, **plot_style_dmiic)
            gbn.plotMetric('time', fig, ax, **plot_style_gbn)

            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] for h in handles]
            ax.legend(handles, labels)
            plt.savefig(os.path.join(bpath,
                                     '_'.join(['time_complexity', distribution , structure]) +'.pdf'),
                        transparent=True)
