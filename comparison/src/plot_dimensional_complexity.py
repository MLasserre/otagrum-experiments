#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')
mpl.rcParams.update({'font.size': 16})

import numpy as np
import pandas as pd
from pathlib import Path
import os
import re

from otagrum import CorrectedMutualInformation as cmi

def combined_mean(m, n):
    m = np.array(m)
    n = np.array(n)
    return np.inner(m,n)/n.sum()

def combined_var(m, v, n):
    m = np.array(m)
    v = np.array(v)
    n = np.array(n)
    cm = combined_mean(m, n)
    mu2 = (m - cm)**2
    product =  np.inner(n, v + mu2)/n.sum()
    return product

def combined_std(m, v, n):
    return np.sqrt(combined_var(m, v, n))

def combined_var_df(m, v):
    n = np.full(m.shape[1], 5)
    cm = m.apply(lambda x:combined_mean(x, np.full(len(x), 5)), axis=1)
    mu2 = (np.subtract(m.T, cm)**2).T
    s = np.add(v, mu2)
    f = s.apply(lambda x:np.inner(x, n), axis=1)
    d = f/n.sum()
    return d

def combined_std_df(m, v):
    return np.sqrt(combined_var_df(m, v))

def param_to_string(parameters):
    return '_'.join([str(parameter).replace('.', '') for parameter in parameters.values()])

# sizes = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 62, 72, 82, 92, 102]
sizes = [2, 7, 12]
# sizes = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52]
# distributions = ["dirichlet"]
distribution = 'gaussian'
methods = {'cpc':{'binNumber':5, 'alpha':0.05},
           'cbic':{'max_parents':4, 'hc_restart':5, 'cmode':cmi.CModeTypes_Gaussian},
           'g-cmiic':{'cmode':cmi.CModeTypes_Gaussian, 'kmode':cmi.KModeTypes_Naive},
           'b-cmiic':{'cmode':cmi.CModeTypes_Bernstein, 'kmode':cmi.KModeTypes_Naive},
           'dmiic':{'dis_method':'quantile', 'nbins':5, 'threshold':25},
           'gbn':{'maxp':4, 'restart':5}}
metrics = ['skelF', 'hamming', 'time']
results_dir = Path("../results")

results = {method:{metric:{'mean':pd.DataFrame(), 'std':pd.DataFrame()}
                   for metric in metrics}
           for method in methods}

for method in methods:
    print('Method: {}'.format(method))
    # Magouille, il faudrait plutot mettre une boucle sur les parametres
    # mais que mettre alors pour le label de la courbe ?
    if method.find('-') != -1:
        pmethod = method.split('-')[1]
    else:
        pmethod = method
    for metric in metrics:
        print('\tMetric: {}'.format(metric))
        for size in sizes:
            print('\t\tSize: {}'.format(size))

            # Need to differenciate because of correlation
            if distribution == 'gaussian' or distribution == 'student':
                partial_csv_paths = 'size_'+ str(size).zfill(3) + '_0*' + '/r08' + \
                                    '/processed/' + metric + '_' + pmethod + \
                                    '_' + param_to_string(methods[method]) + \
                                    '_f*t*np*r5.csv'
            elif distribution == 'dirichlet':
                partial_csv_paths = 'size_' + str(size).zfill(3) + '_0*' + \
                                    '/processed/' + metric + '_' + pmethod + \
                                    '_' + param_to_string(methods[method]) + \
                                    '_f*t*np*r5.csv'
            else:
                print("Distribution needs to be Gaussian, Student or Dirichlet")

            full_csv_paths = \
                    sorted(results_dir.joinpath(distribution).glob(partial_csv_paths))

            by_size_results = {'mean':[], 'std':[]}
            for f in full_csv_paths:
                df = pd.read_csv(f, delimiter=',', header=0)
                df.columns = df.columns.str.strip('# ')
                df['Size'] = df['Size'].astype(int)
                df.set_index('Size', inplace=True)
                by_size_results['mean'].append(df['Mean'])
                by_size_results['std'].append(df['Std'])

            merged_results = {'mean':pd.DataFrame(), 'std':pd.DataFrame()}
            for bsrm in by_size_results['mean']:
                merged_results['mean'] = pd.concat([merged_results['mean'], bsrm],
                                                   axis=1, sort=False)

            for bsrs in by_size_results['std']:
                merged_results['std'] = pd.concat([merged_results['std'], bsrs],
                                               axis=1, sort=False)

            aggregated_results = {}
            # aggregated_results['mean'] = \
                # merged_results['mean'].apply(lambda x:combined_mean(x, np.full(len(x), 5)), axis=1)
            # aggregated_results['std'] = combined_std_df(merged_results['mean'],
                                                        # merged_results['std']**2)
            aggregated_results['mean'] = merged_results['mean'].mean(axis=1).to_frame(size)
            aggregated_results['std'] = merged_results['std'].std(axis=1).to_frame(size)
            results[method][metric]['mean'] = \
                    pd.concat([results[method][metric]['mean'], aggregated_results['mean']],
                              axis=1, sort=False)
            results[method][metric]['std'] = \
                    pd.concat([results[method][metric]['std'], aggregated_results['std']],
                              axis=1, sort=False)

# time_mean.columns = sizes
for method in methods:
    for metric in metrics:
        for key in results[method][metric]:
            results[method][metric][key] = results[method][metric][key].T
            results[method][metric][key].columns = results[method][metric][key].columns.astype(int)

# PLOTTING DIMENSIONAL COMPLEXITIES

colors = ['maroon', 'olivedrab', 'goldenrod', 'royalblue', 'red', 'green']
linestyles = ['-.', (0,(1,1)), '--', '-', '-', '-']
local_plot_style = {method:{'color':c, 'linestyle':ls}
                    for (method, c, ls) in zip(methods, colors,linestyles)}
general_plot_style = {'capsize':2, 'elinewidth':1.25, 'linewidth':1.25, 'loglog':True}

if distribution == 'gaussian' or distribution == 'student':
    fig_directory = Path("../figures")/distribution/"generated"
elif distribution == 'dirichlet':
    fig_directory = Path("../figures")/distribution/"generated"

sample_size = 10000 # Using samples of size 10000 

# PLOTTING DIMENSIONAL TIME COMPLEXITY
fig, ax = plt.subplots()

# ax.set_xlabel('Number of nodes')
# ax.set_ylabel('Time (s)')

#ax.set_title(fig_title)

# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_xlim([start_size, end_size])
# #ax.set_ylim(0.,3.)

def myLogFormat(y,pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)

for method in methods:
    results[method]['time']['mean'][sample_size].plot(
            yerr=results[method]['time']['std'][sample_size],
            label=method, ax=ax,
            **local_plot_style[method],
            **general_plot_style)

ax.set_xscale('log', base=2)
ax.set_yscale('log')

ax.set_xlim([sizes[0], sizes[-1]])
for axis in [ax.xaxis]:
    axis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
# ax.set_ylim(0.,3.)

ax.legend()

plt.savefig(fig_directory.joinpath('time_' + distribution + '_dimensional_complexity.pdf'),
            transparent=True)
print("Saving figure in ", fig_directory.joinpath('time_' + distribution +
                           '_dimensional_complexity.pdf'))

general_plot_style['loglog'] = False

# PLOTTING FSCORE DIMENSIONAL COMPLEXITY
fig, ax = plt.subplots()

# ax.set_xlabel('Number of nodes')
# ax.set_ylabel('F-score')

for method in methods:
    results[method]['skelF']['mean'][sample_size].plot(
            yerr=results[method]['skelF']['std'][sample_size],
            label=method, ax=ax,
            **local_plot_style[method],
            **general_plot_style)
ax.set_xlim([sizes[0], sizes[-1]])
ax.legend()

plt.savefig(fig_directory.joinpath('skelF_' + distribution + '_dimensional_complexity.pdf'),
            transparent=True)
print("Saving figure in ", fig_directory.joinpath('skelF_' + distribution +
                                                  '_dimensional_complexity.pdf'))



# PLOTTING HAMMING DISTANCE DIMENSIONAL COMPLEXITY
fig, ax = plt.subplots()

# ax.set_xlabel('Number of nodes')
# ax.set_ylabel('SHD')

for method in methods:
    results[method]['hamming']['mean'][sample_size].plot(
            yerr=results[method]['hamming']['std'][sample_size],
            label=method, ax=ax,
            **local_plot_style[method],
            **general_plot_style)

ax.set_xlim([sizes[0], sizes[-1]])
ax.legend()

plt.savefig(fig_directory.joinpath('hamming_' + distribution + '_dimensional_complexity.pdf'),
            transparent=True)
print("Saving figure in ", fig_directory.joinpath('hamming_' + distribution +
                                                  '_dimensional_complexity.pdf'))
