# otagrum-experiments

This repository contains all the experiments made during my PhD with the otagrum library 
([openturns/otagrum](https://github.com/openturns/otagrum)).

For now, the experiments are using a development of ([aGrUM](https://gitlab.com/agrumery/aGrUM)) and
as such you will need to install otagrum manually.

For Linux users, create a new repository where you will clone all the necessary libraries:
```bash
mkdir git-repos
cd git-repos
git clone https://gitlab.com/agrumery/aGrUM.git
git clone https://github.com/openturns/openturns.git
git clone https://github.com/openturns/otagrum.git
git clone https://github.com/MLasserre/otagrum-experiments.git
```
First you need to install the development branch of aGrUM:
```bash
cd aGrUM
git checkout feature/mapForStructuralComparator
./act install release aGrUM
./act install release pyAgrum
```

Next, you need to install ([OpenTURNS](https://github.com/openturns/openturns)):
```bash
cd ../openturns/
mkdir build
cd build
cmake ..
make install
```

Finally, you can install otagrum:
```bash
cd ../../otagrum
mkdir build
cd build
cmake ..
make install
```

If you are lucky enough and having no issues, you should now be able to use the scripts to plot figures:
```bash
cd ../../otagrum-experiments/comparison/src/
python3 main.py
```

If you are just interested in using otagrum and to do your own experiments, you can simply install it using ([conda](https://docs.conda.io/en/latest/miniconda.html)):
```bash
conda install -c conda-forge otagrum
```

If you use the conda way, you will not be able to use the experiments here.
Before you can, I need to merge my work into the master of aGrUM.
It should be done soon !
