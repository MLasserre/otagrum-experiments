import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
#mpl.rcParams.update({'font.size': 20})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["royalblue", "darkorange", "olivedrab", "crimson", "mediumpurple", "sienna", "pink"])

red_wine_data = pd.read_csv('../data/winequality-red.csv', delimiter=';')
white_wine_data = pd.read_csv('../data/winequality-white.csv', delimiter=';')

print(white_wine_data.min())
print(white_wine_data.max())
print(white_wine_data.mean())
print(white_wine_data.std())

red_percentage_quality = red_wine_data.quality.value_counts(normalize=True) * 100
red_percentage_quality.sort_index(inplace=True)
white_percentage_quality = white_wine_data.quality.value_counts(normalize=True) * 100
white_percentage_quality.sort_index(inplace=True)

print(red_percentage_quality)

fig, ax = plt.subplots()
red_percentage_quality.plot.bar(ax=ax, edgecolor='black', color='crimson')
white_percentage_quality.plot.bar(ax=ax, edgecolor='black', color='white')
#ax.hist(red_wine_data.quality, bins=11, color='crimson', edgecolor='black', density=True)
#ax.hist(white_wine_data.quality, bins=11, color='white', edgecolor='black', density=True)
ax.set_xlim(0, 10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()