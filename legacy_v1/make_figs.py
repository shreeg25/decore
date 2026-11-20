import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results.json"))
plt.rcParams.update({"font.size": 8, "figure.figsize": (3.4, 2.4), "axes.grid": True, "grid.alpha": 0.3})
EPS = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]


def ms(d):
    a = np.array(d["miss"]); return a.mean(), a.std() / np.sqrt(len(a))


# Fig 1: miss rate vs budget, noise vs spoof, plain vs ens3
P = R["primary"]
fig, ax = plt.subplots()
for pol, ls in [("plain", "-"), ("ens3", "--")]:
    for kind, c in [("noise", "tab:blue"), ("spoof", "tab:red")]:
        m = [ms(P[f"{pol}|{kind}{e}"])[0] for e in EPS]
        s = [ms(P[f"{pol}|{kind}{e}"])[1] for e in EPS]
        ax.errorbar(EPS, m, yerr=s, ls=ls, color=c, marker="o", ms=3, capsize=2,
                    label=f"{pol} / {kind}")
ax.set_xscale("log"); ax.set_xlabel("perturbation budget (slots)"); ax.set_ylabel("deadline-miss rate (%)")
ax.legend(fontsize=6); fig.tight_layout(); fig.savefig("fig1_budget.pdf"); plt.close(fig)

# Fig 2: herd index (modal share) vs budget, plain, spoof vs noise
fig, ax = plt.subplots()
for kind, c in [("noise", "tab:blue"), ("spoof", "tab:red")]:
    m = [np.mean(P[f"plain|{kind}{e}"]["modal"]) for e in EPS]
    ax.plot(EPS, m, color=c, marker="o", ms=3, label=kind)
ax.axhline(np.mean(P["plain|clean"]["modal"]), color="k", ls=":", label="clean")
ax.set_xscale("log"); ax.set_xlabel("perturbation budget (slots)"); ax.set_ylabel("share of tasks to modal server")
ax.legend(fontsize=6); fig.tight_layout(); fig.savefig("fig2_herd.pdf"); plt.close(fig)

# Fig 3: load and device-count sweeps (spoof 0.5, plain vs ens3)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4))
for ax, key, xs, xl in [(axs[0], "rho_sweep", ["0.4", "0.5", "0.6", "0.7"], "utilization"),
                        (axs[1], "n_sweep", ["10", "20", "40"], "number of devices")]:
    for pol, ls in [("plain", "-"), ("ens3", "--")]:
        for cond, c in [("clean", "k"), ("noise0.5", "tab:blue"), ("spoof0.5", "tab:red")]:
            m = [ms(R[key][x][f"{pol}|{cond}"])[0] for x in xs]
            ax.plot([float(x) for x in xs], m, ls=ls, color=c, marker="o", ms=3, label=f"{pol}/{cond}")
    ax.set_xlabel(xl); ax.set_ylabel("deadline-miss rate (%)")
axs[1].legend(fontsize=5)
fig.tight_layout(); fig.savefig("fig3_sweeps.pdf"); plt.close(fig)

# Table: defenses at primary setting
rows = []
for pol in ["plain", "temp2", "ens3", "ens5"]:
    r = [pol, f"{R['latency_ms'][pol]:.3f}"]
    for cond in ["clean", "stale3", "noise0.5", "spoof0.5", "spoof1.0", "spoof2.0"]:
        m, s = ms(P[f"{pol}|{cond}"]); r.append(f"{m:.1f}±{s:.1f}")
    rows.append(r)
hdr = ["policy", "lat ms", "clean", "stale3", "noise0.5", "spoof0.5", "spoof1.0", "spoof2.0"]
with open("table_defenses.md", "w") as f:
    f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
    for r in rows:
        f.write("| " + " | ".join(r) + " |\n")
print(open("table_defenses.md").read())

# Collapse counts: fraction of eval seeds with miss > 20% for plain vs ens3
for cond in ["clean", "noise0.5", "spoof0.5", "spoof1.0"]:
    out = []
    for pol in ["plain", "ens3"]:
        a = np.array(P[f"{pol}|{cond}"]["miss"])
        out.append(f"{pol}: {np.mean(a > 20)*100:.0f}% of runs >20% miss")
    print(cond, "|", " ; ".join(out))
