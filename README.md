# Class-balanced, momentum-accumulated federated learning vs. FedAvg on sparse, scarce clients

## Motivation
Federated learning often has to train on clients whose local data is sparse, scarce and skewed toward a few classes. The paper "Safa: Handling sparse and scarce data in federated learning with accumulative learning" (2025) addresses this with an accumulative-learning approach. This repository is a small, self-contained experiment that asks whether a simple class-balanced, momentum-accumulated federated update helps in that setting, using a standard FedAvg baseline for comparison.

## Question
Does a class-balanced, momentum-accumulated federated update beat a standard FedAvg baseline when clients hold sparse, scarce and non-IID data?

## Dataset
The scikit-learn breast-cancer dataset (bundled with the package, binary label, tabular features), split once into a stratified hold-out train/test set. The training part is partitioned across simulated clients with Dirichlet label skew, small per-client sample counts and random feature masks. This is not the paper's own dataset.

## Baseline vs. improved model
- **Baseline:** standard FedAvg with a logistic-regression model trained by local SGD and averaged by sample count.
- **Improved:** the same local model and data, plus class-balanced local loss weights, a server-side momentum (accumulative) update, and aggregation weights that down-weight clients with very few samples.

Both use the same seeds, rounds and compute budget. The main metric is macro-F1 on the shared test set; accuracy and minority-class recall are secondary.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a comparison chart to `figures/`) when `analysis.py` finishes.

## Reference

This project is motivated by the following paper:

> NH Nguyen, TT Nguyen, TN Hoang (2025). *Safa: Handling sparse and scarce data in federated learning with accumulative learning*. IEEE Transactions. https://ieeexplore.ieee.org/abstract/document/10908581/

Paper author: Prof. Nguyen Thanh Hung, Hanoi University of Science and Technology (hungnt@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
