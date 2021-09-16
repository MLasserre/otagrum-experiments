import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as path

import numpy as np
from scipy.special import binom, beta
from scipy.stats import norm
import otagrum
from openturns import (
        CorrelationMatrix,
        ComposedDistribution,
        EmpiricalBernsteinCopula,
        Normal,
        RandomGenerator,
        SpecFunc,
        Uniform
)
from pathlib import Path
from tqdm import tqdm

def save_dict(dictionary, location='.'):
    path = Path(location)
    path.mkdir(parents=True, exist_ok=True)
    for key in dictionary:
        name = str(key).zfill(7)
        np.savetxt(path/(name+'.csv'), dictionary[key], delimiter=',')

def load_dict(path, sizes):
    path = Path(path)
    dictionary = {}
    files_to_load = sorted([f for f in path.iterdir() if int(f.stem) in sizes],
                           key=lambda f: int(f.stem))
    for f in files_to_load:
        dictionary[int(f.stem)] = np.loadtxt(f, delimiter=',')
        print("\tLoaded file {}".format(f))
    return dictionary
        
def compute_results(sizes, restarts, distribution):
    info_results = {}
    hdist_results = {}
    for size in sizes:
        print(size)
        size = int(size)

        infos = []
        hdists = []
        for _ in tqdm(range(restarts)):
            sample = distribution.getSample(size)
            sample = (sample.rank() + 1) / (sample.getSize() + 2)
            # sample.exportToCSVFile("sample_for_Regis.csv")

            # Computing mutual information using otagrum
            icomputer = otagrum.CorrectedMutualInformation(sample)
            icomputer.setKMode(otagrum.CorrectedMutualInformation.KModeTypes_NoCorr)
            info = icomputer.compute2PtCorrectedInformation(0, 1)

            # Computing hellinger distance using otagrum
            ttest = otagrum.ContinuousTTest(sample, 0.05)
            hdist = ttest.getHDistance(0, 1, [])

            infos.append(info)
            hdists.append(hdist)

        info_results[size] = np.array(infos)
        hdist_results[size] = np.array(hdists)
        
        save_dict(info_results, Path('results')/'info'/str(restarts))
        save_dict(hdist_results, Path('results')/'hdist'/str(restarts))

    return info_results

def apply_to_dic(dictionary, method):
    new = {}
    for key in dictionary:
        new[key] = method(dictionary[key])
    return new


# Main
RandomGenerator.SetSeed(0)
size_min, size_max = 1000, 10001
sizes = np.arange(size_min, size_max, 1000)
restarts = 5000

res_dir = Path('../results')
fig_dir = Path('../figures')

info_dir = res_dir/'info'/str(restarts)
hdist_dir = res_dir/'hdist'/str(restarts)

print(info_dir)

distribution = ComposedDistribution([Uniform(0., 1.)]*2)

print("Checking existing results")
if not info_dir.exists() or not hdist_dir.exists():
    print("Found no results, computing them...")
    info_results = compute_results(sizes, restarts, distribution)
else:
    # Find which size files have been generated
    found_info_sizes = [int(f.stem) for f in info_dir.iterdir()]
    found_hdist_sizes = [int(f.stem) for f in hdist_dir.iterdir()]
    remaining_info_sizes = [size for size in sizes if size not in found_info_sizes]
    remaining_hdist_sizes = [size for size in sizes if size not in found_hdist_sizes]
    remaining_sizes = list(set(remaining_info_sizes + remaining_hdist_sizes))

    # Generate the remaining ones
    if remaining_sizes:
        print("Computing missing results...")
        remaining_sizes.sort()
        compute_results(remaining_sizes, restarts, distribution)

    # Load the results into dictionaries
    print("Loading data...")
    info_results = load_dict(info_dir, sizes)
    hdist_results = load_dict(hdist_dir, sizes)
    print("Done!")
    
for size in sizes:
    plt.hist(info_results[size], bins=150, label=str(size))
plt.legend()
plt.savefig(fig_dir/"info_histogram_{}_{}.pdf".format(size,restarts), transparent=True)
plt.clf()

for size in sizes:
    plt.hist(hdist_results[size], bins=150, label=str(size))
plt.savefig(fig_dir/"hdist_histogram_{}_{}.pdf".format(size,restarts), transparent=True)
plt.legend()
plt.clf()


# Compute mean and std
print("Computing mean and standard deviation...")
means = apply_to_dic(info_results, np.mean)
stds = apply_to_dic(info_results, np.std)
print("Done!")

fig, ax = plt.subplots()
ax.errorbar(sizes, list(means.values()), list(stds.values()))
ax.plot(sizes, list(means.values()))
plt.savefig(fig_dir/"mean_info.pdf", transparent=True)
# plt.show()

fig, ax = plt.subplots()
ax.plot(sizes, list(stds.values()))
plt.savefig(fig_dir/"std_info.pdf", transparent=True)
# plt.show()

print("Computing mean and standard deviation...")
means = apply_to_dic(hdist_results, np.mean)
stds = apply_to_dic(hdist_results, np.std)
print("Done!")

fig, ax = plt.subplots()
ax.errorbar(sizes, list(means.values()), list(stds.values()))
ax.plot(sizes, list(means.values()))
plt.savefig(fig_dir/"mean_hdist.pdf", transparent=True)
# plt.show()

fig, ax = plt.subplots()
ax.plot(sizes, list(stds.values()))
plt.savefig(fig_dir/"std_hdist.pdf", transparent=True)
# plt.show()
