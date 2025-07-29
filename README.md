# Robustness Verification for Ensemble Learning Methods
This repository contains all of the code that was written for the paper "Robustness Verification for Ensemble Learning Methods." The paper was submitted as an Independent Work (IW) project to the Princeton COS department for the Spring 2025 semester and was also used to fulfill the IW requirements for the Statistics and Machine Learning minor. See the `Report/` subdirectory for .pdf and .tex sources of the final written report.

Author: **Samuel Sanft**
Adviser: **Aarti Gupta**

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
