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

## Results (latest run)

On the scikit-learn breast cancer dataset, a standard FedAvg logistic-regression baseline reached 96.6% macro-F1 on the shared held-out test set, while the class-balanced, momentum-accumulated variant reached 96.6%, a negligible difference within seed noise (std about 0.9 points, three seeds).

| Metric | Value |
|---|---|
| baseline_score | 0.9657 |
| baseline_score_std | 0.008575 |
| baseline_accuracy | 0.9678 |
| baseline_minority_recall | 0.9683 |
| improved_score | 0.9658 |
| improved_score_std | 0.008641 |
| improved_accuracy | 0.9678 |
| improved_minority_recall | 0.9762 |
| improvement | 0.00009352 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in baseline/improvement inspired by accumulative learning; the Safa paper's exact update was not re-implemented. Hyperparameters were fixed a priori (no tuning, so no test-set selection). Feature scaler is fit on the pooled training split (a simplification of a real federated setting). The test set is small, so differences are noisy; only three seeds were run. Results do not test the paper's own claims or dataset.

## Reference

This project is motivated by the following paper:

> NH Nguyen, TT Nguyen, TN Hoang (2025). *Safa: Handling sparse and scarce data in federated learning with accumulative learning*. IEEE Transactions. https://ieeexplore.ieee.org/abstract/document/10908581/

Paper author: Prof. Nguyen Thanh Hung, Hanoi University of Science and Technology (hungnt@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
