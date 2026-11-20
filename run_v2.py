"""run_v2.py - herding study v2. Needs the patched herd_lib.py in the same folder.

python run_v2.py            # full run -> results_v2.json
python run_v2.py --quick    # ~2 min smoke test, tiny sample, no sweeps
Run it from a normal terminal (not a call that returns before it finishes).

Regime: deadline D=2 slots, primary load 0.8 (at D=4 the non-learned rules barely react to spoofing, so there is nothing to herd).
Training observations now carry the same information as evaluation observations (lag=0). v1 trained on a
one-slot-older backlog than it was evaluated on (lag=1), which broke the learned policies (43-90% clean miss).
"""
import argparse, json, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import herd_lib as H
from herd_lib import Env, Policy, evaluate, latency_ms, draw_slot, obs_of, expert_action, serve, perturb, M

D_NEW = 2.0
H.D = D_NEW
EPS = [0.25, 0.5, 1.0, 2.0]
GATE_MARGIN = 3.0     # keep a learned model only if clean validation miss <= expert + 3 points
MIN_KEEP = 5


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _plain(N, rho, seed, steps, iters):
    H.D = D_NEW
    return H.train_policy(N, rho, seed, steps=steps, iters=iters)


def _aug(N, rho, seed, steps, iters, p_aug=0.5, emax=1.0):
    """Same as H.train_policy but the broadcast seen by the policy is perturbed (random noise / spoof with a
    random target server, eps~U(0,emax)) in p_aug of the slots. Labels still come from the true backlog."""
    H.D = D_NEW
    env = Env(N, rho)
    rng = np.random.default_rng(seed)
    q = np.zeros(M); q_prev = q.copy()
    X, Y = [], []
    for _ in range(steps):
        active, size, tx = draw_slot(env, rng)
        a = expert_action(env, tx, q, rng)
        seen = q / env.cap          # same information the policy has at evaluation time (no lag)
        if rng.random() < p_aug:
            kind = "spoof" if rng.random() < 0.5 else "noise"
            seen = perturb(seen, kind, rng.uniform(0, emax), int(rng.integers(M)), rng)
        X.append(obs_of(env, size, tx, seen)[active]); Y.append(a[active])
        _, _, q_new = serve(env, active, size, tx, a, q, rng)
        q_prev, q = q, q_new
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=iters, random_state=seed))
    clf.fit(np.vstack(X), np.concatenate(Y))
    return clf


def val_miss(model, env, slots):
    p = Policy([model])
    return float(np.mean([evaluate(p, env, "none", 0.0, seed=100 + s, slots=slots)[0] for s in range(3)]))


def gate(models, env, slots, expert_val):
    v = [val_miss(m, env, slots) for m in models]
    thr = expert_val + GATE_MARGIN
    keep = [i for i, x in enumerate(v) if x <= thr]
    npass = len(keep)
    if len(keep) < MIN_KEEP:                       # fall back to the best MIN_KEEP, and say so in the log/json
        keep = list(np.argsort(v)[:MIN_KEEP])
    return [models[i] for i in keep], v, thr, len(keep), npass


def n_params(m):
    c = m[-1]
    return int(sum(a.size for a in c.coefs_) + sum(b.size for b in c.intercepts_))


def make_pols(plain, aug, K=5, extra=False):
    n, na = len(plain), len(aug)
    p = {
        "random": lambda s: Policy([], "random"),
        "jsq": lambda s: Policy([], "jsq"),
        "pow2": lambda s: Policy([], "pow2"),
        "expert": lambda s: Policy([], "expert_seen"),
        "plain": lambda s: Policy([plain[s % n]]),
        "aug": lambda s: Policy([aug[s % na]]),
        "ens_slot": lambda s: Policy([plain[(s + j) % n] for j in range(K)], "ensemble"),
        "ens_dev": lambda s: Policy([plain[(s + j) % n] for j in range(K)], "ensemble_dev"),
    }
    if extra:   # per-device draw over perturbation-trained models
        p["aug_dev"] = lambda s: Policy([aug[(s + j) % na] for j in range(K)], "ensemble_dev")
    return p


def run_cond(pols, env, conds, seeds, slots):
    out = {}
    for pname, mk in pols.items():
        for label, kind, eps, delay in conds:
            r = [evaluate(mk(s), env, kind, eps, seed=s, delay=delay, slots=slots) for s in seeds]
            out[f"{pname}|{label}"] = {"miss": [x[0] for x in r], "modal": [x[1] for x in r],
                                       "tshare": [x[2] for x in r]}
    return out


def setting(N, rho, n_models, n_seeds, conds, slots, steps, iters, jobs, only=None, tag="", extra=False):
    t = time.time()
    plain = Parallel(n_jobs=jobs)(delayed(_plain)(N, rho, s, steps, iters) for s in range(n_models))
    aug = Parallel(n_jobs=jobs)(delayed(_aug)(N, rho, 1000 + s, steps, iters) for s in range(n_models))
    log(f"{tag} N={N} rho={rho}: trained {2*n_models} models in {time.time()-t:.0f}s")
    env = Env(N, rho)
    expert_val = float(np.mean([evaluate(Policy([], "expert_seen"), env, "none", 0.0, seed=100 + s, slots=slots)[0]
                                for s in range(3)]))
    gp, vp, thr, kp, npp = gate(plain, env, slots, expert_val)
    ga, va, _, ka, npa = gate(aug, env, slots, expert_val)
    log(f"{tag} gate: expert val miss {expert_val:.1f}, threshold {thr:.1f}; PASSED plain {npp}/{n_models}, aug {npa}/{n_models}")
    if min(npp, npa) < MIN_KEEP:
        log(f"{tag} WARNING: fewer than {MIN_KEEP} models passed the gate, kept the best {MIN_KEEP} anyway - train longer (steps/iters) before trusting learned-policy numbers")
    pols = make_pols(gp, ga, extra=extra)
    if only:
        pols = {k: v for k, v in pols.items() if k in only}
    res = run_cond(pols, env, conds, list(range(n_seeds)), slots)
    meta = {"N": N, "rho": rho, "D": D_NEW, "expert_val": expert_val, "thresh": thr,
            "n_trained": n_models, "n_kept_plain": kp, "n_kept_aug": ka, "n_pass_plain": npp, "n_pass_aug": npa,
            "plain_val": vp, "aug_val": va}
    return res, meta, pols, env, gp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--extra", action="store_true", help="adaptive attacker + aug_dev, primary and N=80 only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = "results_extra.json" if a.extra else "results_v2.json"

    if a.quick:
        n_models, n_seeds, slots, steps, iters = 6, 4, 200, 1500, 60
    else:
        n_models, n_seeds, slots, steps, iters = 30, 20, 600, 4000, 150
    conds_full = [("clean", "none", 0.0, 1), ("stale3", "none", 0.0, 3)] + \
                 [(f"{k}{e}", k, e, 1) for k in ("noise", "spoof") for e in EPS]
    conds_small = [("clean", "none", 0.0, 1), ("noise0.5", "noise", 0.5, 1), ("spoof0.5", "spoof", 0.5, 1)]
    sweep_pols = ("random", "jsq", "pow2", "expert", "plain", "aug", "ens_dev")

    if a.extra:
        BUD = [0.5, 1.0, 2.0]
        conds_x = [("clean", "none", 0.0, 1)] + [(f"{k}{e}", k, e, 1) for k in ("noise", "spoof", "adv") for e in BUD]
        res = {"eps": BUD, "D": D_NEW, "gate": {}}
        r1, m1, *_ = setting(20, 0.8, n_models, n_seeds, conds_x, slots, steps, iters, a.jobs, tag="extra-primary", extra=True)
        res["primary"] = r1; res["gate"]["primary"] = m1
        json.dump(res, open(a.out, "w")); log("extra primary done")
        if not a.quick:
            r2, m2, *_ = setting(80, 0.8, 8, 10, conds_x, 400, steps, iters, a.jobs, tag="extra-N80", extra=True)
            res["n80"] = r2; res["gate"]["N80"] = m2
            json.dump(res, open(a.out, "w"))
        log("EXTRA DONE"); return

    res = {"eps": EPS, "D": D_NEW, "gate": {}}

    def save():
        json.dump(res, open(a.out, "w"))

    # primary: N=20, rho=0.8 (random assignment is clearly worse than a good rule here, so the report has value)
    N0, rho0 = 20, 0.8
    prim, meta, pols, env, gp = setting(N0, rho0, n_models, n_seeds, conds_full, slots, steps, iters, a.jobs, tag="primary")
    res["primary"] = prim; res["gate"]["primary"] = meta
    res["latency_ms"] = {k: latency_ms(mk(0), env) for k, mk in pols.items()}
    P = n_params(gp[0]); K = 5
    res["params"] = {"plain": P, "aug": P, "ens_slot": K * P, "ens_dev": K * P, "jsq": 0, "pow2": 0, "expert": 0, "random": 0}
    save(); log("primary done")
    if a.quick:
        log("quick mode: sweeps skipped"); return

    # sweeps: fewer models / seeds / conditions
    res["rho_sweep"] = {}; res["n_sweep"] = {}
    for rho in [0.6, 0.7, 0.9]:
        r, m, *_ = setting(20, rho, 8, 10, conds_small, 400, steps, iters, a.jobs, only=sweep_pols, tag="rho-sweep")
        res["rho_sweep"][str(rho)] = r; res["gate"][f"rho{rho}"] = m; save()
    res["rho_sweep"]["0.8"] = {k: v for k, v in prim.items()
                               if k.split("|")[1] in [c[0] for c in conds_small] and k.split("|")[0] in sweep_pols}
    for n in [40, 80]:
        r, m, *_ = setting(n, 0.8, 8, 10, conds_small, 400, steps, iters, a.jobs, only=sweep_pols, tag="N-sweep")
        res["n_sweep"][str(n)] = r; res["gate"][f"N{n}"] = m; save()
    res["n_sweep"]["20"] = res["rho_sweep"]["0.8"]
    save(); log("ALL DONE")


if __name__ == "__main__":
    main()
