# Robustness and Interpretability of Tree-based Ensemble Learning Methods
Author: **Samuel Sanft**

## Setting up the environment
```
conda env create -n IW25 -f environment.yaml
conda activate IW25
make
```

## Run Tests
```
make runtests
```

## Run Evaluations on California Housing Dataset
```
python -m examples.evaluations.california --all
```
