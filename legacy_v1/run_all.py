import json, sys, time
import numpy as np
from herd_lib import *

OUT = "results.json"
res = {}


def save():
    json.dump(res, open(OUT, "w"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_cond(models, env, pols, conds, seeds):
    """pols: name -> function(seed)->Policy. conds: list of (label, kind, eps, delay)."""
    out = {}
    for pname, mk in pols.items():
        for label, kind, eps, delay in conds:
            r = [evaluate(mk(s), env, kind, eps, seed=s, delay=delay) for s in seeds]
            out[f"{pname}|{label}"] = {"miss": [x[0] for x in r], "modal": [x[1] for x in r]}
    return out


def pols_for(models, K_list=(3, 5)):
    n = len(models)
    p = {"plain": lambda s: Policy([models[s % n]])}
    for K in K_list:
        p[f"ens{K}"] = (lambda K: lambda s: Policy([models[(s + j) % n] for j in range(K)], "ensemble"))(K)
    p["temp2"] = lambda s: Policy([models[s % n]], "temp", T=2.0)
    return p


EPS = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
conds_full = [("clean", "none", 0.0, 1), ("stale3", "none", 0.0, 3)] + \
             [(f"{k}{e}", k, e, 1) for k in ("noise", "spoof") for e in EPS]
conds_small = [("clean", "none", 0.0, 1), ("stale3", "none", 0.0, 3),
               ("noise0.5", "noise", 0.5, 1), ("spoof0.5", "spoof", 0.5, 1)]

# ---- primary: N=20, rho=0.6, 10 models, 20 eval seeds ----
N, rho = 20, 0.6
t = time.time()
models = [train_policy(N, rho, s) for s in range(10)]
log(f"primary models trained in {time.time()-t:.0f}s")
env = Env(N, rho)
seeds = list(range(20))
res["primary"] = run_cond(models, env, pols_for(models), conds_full, seeds)
res["latency_ms"] = {name: latency_ms(mk(0), env) for name, mk in pols_for(models).items()}
res["latency_ms"]["ens1"] = res["latency_ms"]["plain"]
save(); log("primary done")

# ---- load sweep: rho in {0.4,0.5,0.7}, 5 models ----
res["rho_sweep"] = {"0.6": {k: v for k, v in res["primary"].items()
                            if k.split("|")[1] in [c[0] for c in conds_small]}}
for r_ in [0.4, 0.5, 0.7]:
    ms = [train_policy(N, r_, s) for s in range(5)]
    e = Env(N, r_)
    p = {k: v for k, v in pols_for(ms, K_list=(3,)).items() if k in ("plain", "ens3")}
    res["rho_sweep"][str(r_)] = run_cond(ms, e, p, conds_small, list(range(10)))
    save(); log(f"rho {r_} done")

# ---- device-count sweep at rho 0.6: N in {10, 40}, 5 models ----
res["n_sweep"] = {"20": res["rho_sweep"]["0.6"]}
for n_ in [10, 40]:
    ms = [train_policy(n_, 0.6, s) for s in range(5)]
    e = Env(n_, 0.6)
    p = {k: v for k, v in pols_for(ms, K_list=(3,)).items() if k in ("plain", "ens3")}
    res["n_sweep"][str(n_)] = run_cond(ms, e, p, conds_small, list(range(10)))
    save(); log(f"N {n_} done")
log("ALL DONE")
