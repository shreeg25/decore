"""python make_figs_extra.py > stats_extra.txt   (needs results_extra.json from `python run_v2.py --extra`)"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results_extra.json"))
BUD = R["eps"]
POL = ["random", "jsq", "pow2", "expert", "plain", "aug", "ens_slot", "ens_dev", "aug_dev"]
STY = {"random": ("tab:gray", ":"), "pow2": ("tab:green", "-"), "expert": ("k", "-"), "plain": ("tab:red", "-"),
       "aug": ("tab:purple", "--"), "ens_dev": ("tab:blue", "-"), "aug_dev": ("tab:orange", "-")}
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})


def ms(x):
    a = np.array(x, float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)


def paired(a, b, B=5000, seed=0):
    d = np.array(a, float) - np.array(b, float)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d)).mean() for _ in range(B)]
    return d.mean(), np.percentile(bs, [2.5, 97.5])


def report(S, name):
    print(f"\n=== {name} ===")
    cols = ["clean"] + [f"{k}{e}" for e in BUD for k in ("noise", "spoof", "adv")]
    print("| policy | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    for p in POL:
        row = []
        for c in cols:
            m, s = ms(S[f"{p}|{c}"]["miss"]); row.append(f"{m:.1f}±{s:.1f}")
        print(f"| {p} | " + " | ".join(row) + " |")

    def show(label, a, b):
        d, ci = paired(S[a]["miss"], S[b]["miss"])
        print(f"  {label}: {d:+.1f} pts  CI [{ci[0]:.1f}, {ci[1]:.1f}]")

    print("adaptive minus fixed-target spoof (positive = adaptive attacker is stronger)")
    for p in ("expert", "pow2", "plain", "aug", "ens_dev", "aug_dev"):
        for e in BUD:
            show(f"{p} @ {e}", f"{p}|adv{e}", f"{p}|spoof{e}")
    print("defense comparisons, first minus second (positive = second is better)")
    for kind in ("spoof", "adv"):
        for e in BUD:
            show(f"{kind}{e}: aug - aug_dev", f"aug|{kind}{e}", f"aug_dev|{kind}{e}")
            show(f"{kind}{e}: ens_dev - aug_dev", f"ens_dev|{kind}{e}", f"aug_dev|{kind}{e}")
            show(f"{kind}{e}: plain - aug_dev", f"plain|{kind}{e}", f"aug_dev|{kind}{e}")


def panel(ax, S, title):
    for p, (c, ls) in STY.items():
        m = [ms(S[f"{p}|adv{e}"]["miss"])[0] for e in BUD]
        s = [ms(S[f"{p}|adv{e}"]["miss"])[1] for e in BUD]
        ax.errorbar(BUD, m, yerr=s, color=c, ls=ls, marker="o", ms=3, capsize=2, label=p)
    ax.set_xscale("log"); ax.set_xticks(BUD); ax.set_xticklabels([str(b) for b in BUD])
    ax.set_xlabel("perturbation budget (slots)"); ax.set_title(title, fontsize=8)


sets = [("primary", "N=20, load 0.8")] + ([("n80", "N=80, load 0.8")] if "n80" in R else [])
fig, axs = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 2.5), sharey=False, squeeze=False)
for ax, (key, title) in zip(axs[0], sets):
    report(R[key], title); panel(ax, R[key], title)
axs[0][0].set_ylabel("deadline-miss rate (%) under adaptive attack"); axs[0][-1].legend(fontsize=5, ncol=2)
fig.tight_layout(); fig.savefig("fig4_adaptive.pdf"); plt.close(fig)
g = R["gate"]
print("\ngate:", {k: (v["n_pass_plain"], v["n_pass_aug"], v["n_trained"]) for k, v in g.items()})
