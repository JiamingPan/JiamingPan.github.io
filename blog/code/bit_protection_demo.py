"""
Small, reproducible demonstration for "Where would you spend your last bit?"

A transparent sanity check of the idea that WHICH weights you protect under a fixed
precision budget matters, and that ranking weights by how much they move the model's
OUTPUT distribution (a KL / softmax-Fisher score) beats ranking them by weight magnitude
or by loss-gradient.

Setup: multinomial logistic regression (softmax) on sklearn's 8x8 handwritten digits.
Quantize the weight matrix to a coarse uniform grid, but keep the top 10% of weights at
full precision. Compare three ways of choosing that 10%. Averaged over 8 seeds.

Run:  python bit_protection_demo.py   (needs numpy + scikit-learn)
"""

import numpy as np
from sklearn.datasets import load_digits


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(1, keepdims=True)


def run(seed, levels=7, budget=0.10):
    rng = np.random.default_rng(seed)
    d = load_digits()
    X, y = d.data / 16.0, d.target
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
    n = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:n], X[n:], y[:n], y[n:]
    K = int(y.max()) + 1

    # bias as an extra all-ones feature
    Xtr1 = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte1 = np.hstack([Xte, np.ones((len(Xte), 1))])
    W = np.zeros((Xtr1.shape[1], K))
    Ytr = np.eye(K)[ytr]
    for _ in range(3000):                       # plain full-batch GD
        P = softmax(Xtr1 @ W)
        W -= 0.5 * Xtr1.T @ (P - Ytr) / len(Xtr1)

    def acc(Wm):
        return (softmax(Xte1 @ Wm).argmax(1) == yte).mean()

    def mean_kl(Wm):                            # KL( full-precision || candidate ) on test
        P0, P1 = softmax(Xte1 @ W), softmax(Xte1 @ Wm)
        return np.mean(np.sum(P0 * (np.log(P0 + 1e-12) - np.log(P1 + 1e-12)), 1))

    c = np.abs(W).max(); step = 2 * c / (levels - 1)
    q = lambda Wm: np.round(Wm / step) * step  # coarse uniform quantizer
    rerr = W - q(W)                            # rounding error each weight would take

    P = softmax(Xtr1 @ W)
    fj = (P * (1 - P)).mean(0)                 # softmax Fisher diagonal per class
    xi2 = (Xtr1 ** 2).mean(0)                  # feature energy
    s_mag = np.abs(W)                          # protect biggest weights
    s_grad = np.abs(Xtr1.T @ (P - Ytr)) / len(Xtr1) * (rerr ** 2)   # loss-gradient
    s_fish = (xi2[:, None] * fj[None, :]) * (rerr ** 2)             # output-KL / Fisher

    n_keep = int(budget * W.size)
    def protect(score):
        thr = np.sort(score.ravel())[::-1][n_keep - 1]
        keep = score >= thr
        return np.where(keep, W, q(W))         # keep top-budget at full precision

    return {
        "base": acc(W),
        "all":  (acc(q(W)),          mean_kl(q(W))),
        "mag":  (acc(protect(s_mag)),  mean_kl(protect(s_mag))),
        "grad": (acc(protect(s_grad)), mean_kl(protect(s_grad))),
        "fish": (acc(protect(s_fish)), mean_kl(protect(s_fish))),
    }


if __name__ == "__main__":
    R = [run(s) for s in range(8)]
    def agg(key, i=None):
        v = [r[key] if i is None else r[key][i] for r in R]
        return f"{np.mean(v)*100:5.1f}%" if (i == 0 or key == "base") else f"{np.mean(v):.3f}"
    print(f"full precision          acc {agg('base')}")
    print(f"quantize all            acc {agg('all',0)}   meanKL {agg('all',1)}")
    print(f"protect magnitude       acc {agg('mag',0)}   meanKL {agg('mag',1)}")
    print(f"protect loss-gradient   acc {agg('grad',0)}   meanKL {agg('grad',1)}")
    print(f"protect output-KL/Fisher acc {agg('fish',0)}  meanKL {agg('fish',1)}")
