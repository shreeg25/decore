"""Herding study: learned offloading policy under stale / noisy / spoofed broadcast load reports.

numpy + scikit-learn only. N devices, M servers, slotted time. Each device observes its task size,
its transmission delay to each server and a BROADCAST backlog vector (previous slot, in slots).
Policies are MLPs trained by imitating a stochastic expert that sees the true backlog.
"""
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

M = 4
P_ARR = 0.5
D = 4.0
TAU = 0.5


class Env:
    def __init__(self, N, rho):
        self.N, self.rho = N, rho
        self.cap = N * P_ARR * 1.0 / (M * rho)


def draw_slot(env, rng):
    N = env.N
    active = rng.random(N) < P_ARR
    size = rng.uniform(0.5, 1.5, N)
    g = rng.lognormal(0.0, 0.3, (N, M))
    tx = 0.05 * size[:, None] / g
    return active, size, tx


def obs_of(env, size, tx, load_seen):
    return np.concatenate([size[:, None], tx, np.tile(load_seen, (env.N, 1))], axis=1)


def sample_rows(pr, rng):
    c = pr.cumsum(1)
    return (rng.random((len(pr), 1)) > c).sum(1).clip(0, M - 1)


def expert_action(env, tx, q_true, rng):
    d = tx + q_true[None, :] / env.cap
    w = np.exp(-(d - d.min(1, keepdims=True)) / TAU)
    w /= w.sum(1, keepdims=True)
    return sample_rows(w, rng)


def serve(env, active, size, tx, act, q, rng):
    cap = env.cap
    miss, n = 0, int(active.sum())
    new_q = q.copy()
    for m in range(M):
        idx = np.flatnonzero(active & (act == m))
        if len(idx) == 0:
            continue
        ahead = q[m]
        for i in rng.permutation(idx):
            if tx[i, m] + ahead / cap + size[i] / cap > D:
                miss += 1
            ahead += size[i]
        new_q[m] = ahead
    return miss, n, np.maximum(0.0, new_q - cap)


def train_policy(N, rho, seed, steps=4000, iters=150, lag=0):
    env = Env(N, rho)
    rng = np.random.default_rng(seed)
    q = np.zeros(M); q_prev = q.copy()
    X, Y = [], []
    for _ in range(steps):
        active, size, tx = draw_slot(env, rng)
        a = expert_action(env, tx, q, rng)
        X.append(obs_of(env, size, tx, (q_prev if lag else q) / env.cap)[active]); Y.append(a[active])
        _, _, q_new = serve(env, active, size, tx, a, q, rng)
        q_prev, q = q, q_new
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=iters, random_state=seed))
    clf.fit(np.vstack(X), np.concatenate(Y))
    return clf


class Policy:
    """mode: plain | ensemble (random member per slot) | temp (flattened sampling, T>1)"""
    def __init__(self, models, mode="plain", T=2.0):
        self.models, self.mode, self.T = models, mode, T

    def act(self, X, rng):
        if self.mode in ("jsq", "pow2", "expert_seen", "random"):
            return self._rule(X, rng)
        if self.mode == "ensemble_dev":   # each device draws its own member -> decorrelates within a slot
            idx = rng.integers(len(self.models), size=len(X))
            out = np.zeros(len(X), dtype=int)
            for j, m in enumerate(self.models):
                sel = idx == j
                if sel.any():
                    out[sel] = sample_rows(m.predict_proba(X[sel]), rng)
            return out
        if self.mode == "ensemble":
            m = self.models[rng.integers(len(self.models))]
            return sample_rows(m.predict_proba(X), rng)
        pr = self.models[0].predict_proba(X)
        if self.mode == "temp":
            pr = np.maximum(pr, 1e-12) ** (1.0 / self.T)
            pr /= pr.sum(1, keepdims=True)
        return sample_rows(pr, rng)


def _rule(self, X, rng):
    """Non-learned rules acting on the SAME (possibly spoofed) broadcast the learned policies see.
    obs columns: [size, tx(M), load_seen(M)]; load_seen is backlog/cap in slots."""
    d = X[:, 1:1 + M] + X[:, 1 + M:1 + 2 * M]
    if self.mode == "random":
        return rng.integers(M, size=len(X))
    if self.mode == "jsq":
        return d.argmin(1)
    if self.mode == "pow2":
        a = rng.integers(M, size=len(X)); b = rng.integers(M, size=len(X))
        da, db = d[np.arange(len(X)), a], d[np.arange(len(X)), b]
        return np.where(da <= db, a, b)
    w = np.exp(-(d - d.min(1, keepdims=True)) / TAU)      # expert_seen: the training expert, but fed the broadcast
    w /= w.sum(1, keepdims=True)
    return sample_rows(w, rng)


Policy._rule = _rule


def perturb(seen, kind, eps, target, rng):
    l = seen.copy()
    if kind == "spoof":      # target looks lighter, others heavier; L-inf = eps
        l = l + eps
        l[target] = max(0.0, seen[target] - eps)
    elif kind == "noise":    # random sign, same L-inf budget
        l = l + rng.choice([-eps, eps], size=M)
    return np.maximum(0.0, l)


def evaluate(policy, env, kind="none", eps=0.0, seed=0, slots=600, delay=1):
    rng = np.random.default_rng(10_000 + seed)
    q = np.zeros(M)
    hist = [q.copy() for _ in range(delay + 1)]
    miss = tot = 0
    modal = []; tshare = []
    for _ in range(slots):
        active, size, tx = draw_slot(env, rng)
        seen = hist[-delay] / env.cap
        if kind != "none":
            seen = perturb(seen, kind, eps, 0, rng)
        act = np.zeros(env.N, dtype=int)
        if active.any():
            act[active] = policy.act(obs_of(env, size, tx, seen)[active], rng)
        mi, n, q = serve(env, active, size, tx, act, q, rng)
        hist.append(q.copy()); hist.pop(0)
        miss += mi; tot += n
        if n:
            modal.append(np.bincount(act[active], minlength=M).max() / n)
            tshare.append(np.mean(act[active] == 0))
    return 100.0 * miss / max(tot, 1), float(np.mean(modal)), float(np.mean(tshare))


def latency_ms(policy, env, reps=400):
    rng = np.random.default_rng(0)
    active, size, tx = draw_slot(env, rng)
    X = obs_of(env, size, tx, np.zeros(M))[:1]
    for _ in range(20):
        policy.act(X, rng)
    t = time.perf_counter()
    for _ in range(reps):
        policy.act(X, rng)
    return 1000 * (time.perf_counter() - t) / reps
