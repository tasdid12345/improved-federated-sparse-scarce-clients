"""FedAvg baseline vs. class-balanced, server-momentum federated logistic regression
on the bundled breast-cancer data, with sparse, scarce, non-IID simulated clients."""
import json
import sys

import matplotlib
matplotlib.use("Agg")  # headless plotting for GitHub Actions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]          # both methods are run with each seed
N_CLIENTS = 10
DIRICHLET_ALPHA = 0.5         # label skew: smaller alpha = more non-IID
MIN_CLIENT = 5                # every client must hold at least this many samples
KEEP_FRAC = 0.6               # each client observes this fraction of features (rest zeroed -> sparse)
ROUNDS = 30                   # identical compute budget for both methods
LOCAL_EPOCHS = 5
LR = 0.1
BATCH = 8
L2 = 1e-3
SERVER_BETA = 0.7             # improved only: server momentum (accumulative update)
SERVER_LR = 1.0
SHRINK_K = 20                 # improved only: aggregation weight n * n/(n+K) down-weights tiny clients
SOURCE_URL = "https://scikit-learn.org/stable/datasets/toy_dataset.html"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def partition(y, rng):
    """Dirichlet label-skew split of training indices over clients; redraw until all clients are large enough."""
    for _ in range(1000):
        idx = [[] for _ in range(N_CLIENTS)]
        for c in np.unique(y):
            ids = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet(DIRICHLET_ALPHA * np.ones(N_CLIENTS))
            cuts = (np.cumsum(props) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                idx[k].extend(part.tolist())
        if min(len(i) for i in idx) >= MIN_CLIENT:
            return [np.array(i) for i in idx]
    sys.exit("could not build a valid client partition")


def make_clients(X, y, rng):
    """Build per-client (features+bias, labels) with a random feature mask (sparse local data)."""
    clients = []
    for ids in partition(y, rng):
        keep = np.zeros(X.shape[1])
        keep[rng.permutation(X.shape[1])[: int(KEEP_FRAC * X.shape[1])]] = 1.0
        Xk = np.hstack([X[ids] * keep, np.ones((len(ids), 1))])
        clients.append((Xk, y[ids]))
    return clients


def local_train(w, Xk, yk, sw, rng):
    """Mini-batch SGD on (optionally sample-weighted) logistic loss."""
    w = w.copy()
    for _ in range(LOCAL_EPOCHS):
        order = rng.permutation(len(yk))
        for s in range(0, len(yk), BATCH):
            b = order[s:s + BATCH]
            g = sw[b] * (sigmoid(Xk[b] @ w) - yk[b])
            w -= LR * (Xk[b].T @ g / len(b) + L2 * w)
    return w


def run(method, clients, Xte, yte, minority, seed):
    """Train one federated method and evaluate on the shared test set."""
    rng = np.random.default_rng(seed)
    improved = method == "improved"
    sws = []
    for Xk, yk in clients:
        if improved:  # class-balanced local loss weights: n / (2 * n_class)
            cnt = np.bincount(yk, minlength=2).astype(float)
            sws.append(len(yk) / (2.0 * np.maximum(cnt[yk], 1.0)))
        else:
            sws.append(np.ones(len(yk)))
    n = np.array([len(yk) for _, yk in clients], dtype=float)
    a = n * n / (n + SHRINK_K) if improved else n  # improved: shrink weight of tiny clients
    a = a / a.sum()
    w = np.zeros(Xte.shape[1])
    v = np.zeros_like(w)
    for _ in range(ROUNDS):
        ws = [local_train(w, Xk, yk, sw, rng) for (Xk, yk), sw in zip(clients, sws)]
        avg = np.sum([ai * wi for ai, wi in zip(a, ws)], axis=0)
        if improved:  # server momentum on the aggregated update
            v = SERVER_BETA * v + (avg - w)
            w = w + SERVER_LR * v
        else:  # plain FedAvg
            w = avg
    pred = (sigmoid(Xte @ w) >= 0.5).astype(int)
    return {"macro_f1": f1_score(yte, pred, average="macro"),
            "accuracy": accuracy_score(yte, pred),
            "minority_recall": recall_score(yte, pred, pos_label=minority)}


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    # One stratified 80/20 split; the test set is used only at the end.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xte_b = np.hstack([Xte, np.ones((len(Xte), 1))])
    minority = int(np.argmin(np.bincount(ytr)))

    res = {"baseline": [], "improved": []}
    for seed in SEEDS:
        clients = make_clients(Xtr, ytr, np.random.default_rng(seed))
        print(f"seed {seed}: client sizes {[len(yk) for _, yk in clients]}")
        for m in res:
            r = run(m, clients, Xte_b, yte, minority, seed)
            res[m].append(r)
            print(f"  {m}: macro-F1={r['macro_f1']:.4f} acc={r['accuracy']:.4f} "
                  f"minority-recall={r['minority_recall']:.4f}")

    def arr(m, k):
        return np.array([r[k] for r in res[m]])

    metrics = {}
    for m in res:
        metrics[f"{m}_score"] = float(arr(m, "macro_f1").mean())
        metrics[f"{m}_score_std"] = float(arr(m, "macro_f1").std())
        metrics[f"{m}_accuracy"] = float(arr(m, "accuracy").mean())
        metrics[f"{m}_minority_recall"] = float(arr(m, "minority_recall").mean())
    metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
    vals = [v for v in metrics.values()]
    if not np.all(np.isfinite(vals)) or not (0 <= metrics["baseline_score"] <= 1 and 0 <= metrics["improved_score"] <= 1):
        sys.exit("non-finite or out-of-range metrics; failing")

    # Optional chart: baseline vs improved macro-F1 across seeds.
    import os
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.bar(["FedAvg", "Improved"], [metrics["baseline_score"], metrics["improved_score"]],
            yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]], capsize=6,
            color=["grey", "steelblue"])
    plt.ylabel("Macro-F1 (test)")
    plt.title("Baseline vs improved (mean ± std over seeds)")
    plt.tight_layout()
    plt.savefig("figures/macro_f1_comparison.png", dpi=120)
    plt.close()

    out = {
        "question": "Does a class-balanced, momentum-accumulated federated update beat a standard FedAvg baseline "
                    "when clients hold sparse, scarce and non-IID data?",
        "method": "Simulated federated learning on the bundled breast-cancer dataset: Dirichlet label-skew "
                  "partition, random feature masks per client, identical local SGD and round budget; "
                  "both methods evaluated on one shared stratified hold-out test set over several seeds.",
        "dataset": {
            "name": "scikit-learn breast cancer (load_breast_cancer)",
            "source_url": SOURCE_URL,
            "n_rows": int(len(y)),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 on shared test set",
        "higher_is_better": True,
        "baseline": "Standard FedAvg with a logistic-regression model trained by local SGD, averaged by sample count.",
        "improved": "Same local model and data, plus class-balanced local loss weights, server-side momentum "
                    "(accumulative) update, and aggregation weights that down-weight clients with very few samples.",
        "metrics": metrics,
        "notes": "Stand-in baseline/improvement inspired by accumulative learning; the Safa paper's exact update "
                 "was not re-implemented. Hyperparameters were fixed a priori (no tuning, so no test-set selection). "
                 "Feature scaler is fit on the pooled training split (a simplification of a real federated setting). "
                 "The test set is small, so differences are noisy; only three seeds were run. "
                 "Results do not test the paper's own claims or dataset.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
