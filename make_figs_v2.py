import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results_v2.json"))
P, EPS = R["primary"], R["eps"]
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})
STY = {"random": ("tab:gray", ":"), "jsq": ("tab:orange", "-"), "pow2": ("tab:green", "-"),
       "expert": ("k", "-"), "plain": ("tab:red", "-"), "aug": ("tab:purple", "--"),
       "ens_slot": ("tab:blue", ":"), "ens_dev": ("tab:blue", "-")}


def ms(x):
    a = np.array(x, float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)


def paired(a, b, B=5000, seed=0):
    d = np.array(a, float) - np.array(b, float)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d)).mean() for _ in range(B)]
    return d.mean(), np.percentile(bs, [2.5, 97.5])


def curve(ax, kind, field="miss", pols=STY):
    for p, (c, ls) in pols.items():
        m = [ms(P[f"{p}|{kind}{e}"][field])[0] for e in EPS]
        s = [ms(P[f"{p}|{kind}{e}"][field])[1] for e in EPS]
        ax.errorbar(EPS, m, yerr=s, color=c, ls=ls, marker="o", ms=3, capsize=2, label=p)
    ax.set_xscale("log"); ax.set_xlabel("perturbation budget (slots)")


# Fig 1: miss rate vs budget, noise and spoof
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
curve(axs[0], "noise"); curve(axs[1], "spoof")
axs[0].set_title("random noise on shared report", fontsize=8); axs[1].set_title("spoofing rule", fontsize=8)
axs[0].set_ylabel("deadline-miss rate (%)"); axs[1].legend(fontsize=5, ncol=2)
fig.tight_layout(); fig.savefig("fig1_budget.pdf"); plt.close(fig)

# Fig 2: share of tasks sent to the spoofed target server
fig, ax = plt.subplots(figsize=(3.4, 2.4))
curve(ax, "spoof", "tshare")
ax.set_ylabel("share of tasks to target server"); ax.legend(fontsize=5, ncol=2)
fig.tight_layout(); fig.savefig("fig2_target_share.pdf"); plt.close(fig)

# Fig 3: sweeps (if present)
if "rho_sweep" in R:
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax, key, xs, xl in [(axs[0], "rho_sweep", ["0.6", "0.7", "0.8", "0.9"], "utilization"),
                            (axs[1], "n_sweep", ["20", "40", "80"], "number of devices")]:
        for p in ("jsq", "pow2", "plain", "aug", "ens_dev"):
            c, ls = STY[p]
            m = [ms(R[key][x][f"{p}|spoof0.5"]["miss"])[0] for x in xs]
            ax.plot([float(x) for x in xs], m, color=c, ls=ls, marker="o", ms=3, label=p)
        ax.set_xlabel(xl); ax.set_ylabel("miss rate under spoof 0.5 (%)")
    axs[1].legend(fontsize=5)
    fig.tight_layout(); fig.savefig("fig3_sweeps.pdf"); plt.close(fig)

# Table: primary setting
conds = ["clean", "stale3", "noise0.5", "spoof0.5", "spoof1.0", "spoof2.0"]
hdr = ["policy", "lat ms", "params"] + conds
with open("table_v2.md", "w") as f:
    f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
    for p in STY:
        r = [p, f"{R['latency_ms'][p]:.3f}", str(R["params"][p])]
        for c in conds:
            m, s = ms(P[f"{p}|{c}"]["miss"]); r.append(f"{m:.1f}±{s:.1f}")
        f.write("| " + " | ".join(r) + " |\n")
print(open("table_v2.md").read())

# gate report
g = R["gate"]["primary"]
print(f"gate: expert val miss {g['expert_val']:.1f}, thr {g['thresh']:.1f}; passed plain {g['n_pass_plain']}/{g['n_trained']}, aug {g['n_pass_aug']}/{g['n_trained']}")


def per_seed_mean(p, kind):
    return np.mean([np.array(P[f"{p}|{kind}{e}"]["miss"]) for e in EPS], axis=0)


# paired comparisons, mean difference in points with 95% bootstrap CI over eval seeds
def show(name, a, b):
    d, ci = paired(a, b)
    print(f"{name}: {d:+.1f} pts  CI [{ci[0]:.1f}, {ci[1]:.1f}]")


print("\nshared-signal corruption: spoof minus noise, pooled over budgets")
for p in ("jsq", "expert", "pow2", "plain"):
    show(f"  {p}", per_seed_mean(p, "spoof"), per_seed_mean(p, "noise"))
print("noise minus clean at 0.5 (does random noise alone herd?)")
for p in ("jsq", "expert", "pow2", "plain"):
    show(f"  {p}", P[f"{p}|noise0.5"]["miss"], P[f"{p}|clean"]["miss"])
print("defenses at spoof 0.5 (positive = damage removed)")
show("  jsq - pow2", P["jsq|spoof0.5"]["miss"], P["pow2|spoof0.5"]["miss"])
show("  plain - ens_slot", P["plain|spoof0.5"]["miss"], P["ens_slot|spoof0.5"]["miss"])
show("  plain - ens_dev", P["plain|spoof0.5"]["miss"], P["ens_dev|spoof0.5"]["miss"])
show("  ens_slot - ens_dev", P["ens_slot|spoof0.5"]["miss"], P["ens_dev|spoof0.5"]["miss"])
show("  plain - aug", P["plain|spoof0.5"]["miss"], P["aug|spoof0.5"]["miss"])
show("  plain - aug (spoof 2.0, outside aug training range)", P["plain|spoof2.0"]["miss"], P["aug|spoof2.0"]["miss"])
show("  aug - ens_dev (spoof 2.0)", P["aug|spoof2.0"]["miss"], P["ens_dev|spoof2.0"]["miss"])
