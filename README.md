# DECORE: decorrelated offloading under corrupted load reports

Simulation study of herding when many independent offloading policies read the same (stale, noisy or spoofed)
broadcast of edge-server load. N devices, M=4 servers, slotted time, deadline D=2 slots, primary load 0.8.

## Files
- `herd_lib.py`  simulator, imitation-trained MLP policies, non-learned rules (JSQ, power-of-2, expert-on-broadcast,
  random), per-slot and per-device ensembles, attack models (noise, spoof), evaluation.
- `run_v2.py`  full experiment runner (primary setting, load sweep, device-count sweep, validation gate).
- `make_figs_v2.py`  figures, table and paired bootstrap CIs from `results_v2.json`.
- `legacy_v1/`  first version (D=4, no baselines) and its `results.json`. Kept for the record; the D=4 regime is too
  lenient for non-learned rules to react to spoofing. Run from inside that folder with `PYTHONPATH=..`.

## Run
    pip install -r requirements.txt
    python run_v2.py --quick --jobs 1      # ~1 min smoke test, ignore its numbers
    python run_v2.py --jobs -1             # full run -> results_v2.json (run in a terminal that stays open)
    python make_figs_v2.py > stats.txt     # fig1_budget.pdf, fig2_target_share.pdf, fig3_sweeps.pdf, table_v2.md

Check the log line `gate: ... PASSED plain x/30, aug y/30` first. If either is under ~10 the learned models are
undertrained: raise steps/iters in `main()` (e.g. 12000 / 300) and rerun before reading learned-policy results.

## Notes
- v1 trained the policy on a backlog report one slot older than the one it saw at evaluation (`lag=1`), which gave
  43-90% clean miss at 80% load. Fixed: `train_policy(..., lag=0)` is now the default, and the perturbation-trained
  policy uses the same convention. Pass `lag=1` to `train_policy` to reproduce the v1 numbers in `legacy_v1/`.
- Latencies are desktop-CPU, single-decision timings, useful only as relative numbers. Model memory is K x params.
- For the double-blind submission, do not link this repository.
