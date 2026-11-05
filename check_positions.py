"""How well does the evaluator agree with what humans played? Also grid-searches the weights.

Usage: python check_positions.py [--n 40] [--seeds 16] [--cache FILE.pkl] [--grid]

Samples --n random replay positions (fixed seed) plus the 3 development positions, reads both
players' actual turn from the log, and ranks each human choice by EV against the opponent's
equilibrium mix. "worse" = share of our options ranked strictly above the human's (ties count
half): 0 = human played our top option, 0.5 = middle of the pack.
"""

import argparse
import json
import os
import pickle
import random

import numpy as np
import ninjask

import analyze as a

DEV = [("gen9championsvgc2026regmb-2635876837", 3), ("gen9championsvgc2026regmb-2635668924", 1),
       ("gen9championsvgc2026regmb-2635871725", 4)]


def human_turn(log, turn):
    """{'p1a': ('move', 'Heat Wave', 'p2b' or None, mega) | ('switch', 'Sinistcha'), ...}, None if a slot is unknown."""
    i = log.index(f"|turn|{turn}")
    end = next((k for k in range(i + 1, len(log)) if log[k].startswith(("|upkeep", "|turn|", "|win|"))), len(log))
    acts, megas = {}, set()
    for l in log[i + 1:end]:
        w = l.split("|")
        if len(w) < 3:
            continue
        slot = w[2][:3]
        if w[1] == "-mega":
            megas.add(slot)
        elif w[1] == "move" and slot not in acts and "[from]" not in l:
            tgt = w[4][:3] if len(w) > 4 and w[4][:3] in ("p1a", "p1b", "p2a", "p2b") else None
            acts[slot] = ["move", w[3], tgt]
        elif w[1] == "switch" and slot not in acts:
            acts[slot] = ["switch", w[3].split(",")[0]]
    if len(acts) != 4:
        return None
    for s in megas:
        if acts[s][0] == "move":
            acts[s].append(True)
    return acts


def matches(battle, side, choices, acts):
    """Indices of joint choices consistent with the human's logged actions for `side`."""
    out = []
    for k, c in enumerate(choices):
        ok = True
        for slot, part in zip("ab", a.name_choice(battle, side, c).split(" | ")):
            act = acts[side + slot]
            if act[0] == "switch":
                ok &= part == f"switch {act[1]}"
            else:
                name = part.split(" -> ")[0].replace(" (mega)", "")
                ok &= name == act[1] and (("(mega)" in part) == (len(act) > 3))
                if " -> " in part:
                    ok &= act[2] is None or part.split(" -> ")[1][:3] == act[2]  # None: move failed, target unlogged
        if ok:
            out.append(k)
    return out


def worse_share(ev, idx):
    """Share of options strictly better than the best of the human's matching options (ties count half)."""
    h = max(ev[idx])
    better, ties = (ev > h + 1e-9).sum(), (abs(ev - h) <= 1e-9).sum() - 1
    return (better + 0.5 * ties) / (len(ev) - 1)


def sample_positions(logs, n, rng):
    ids = sorted(k for k in logs if k not in dict(DEV))
    out = []
    while len(out) < n:
        rid = rng.choice(ids)
        log = logs[rid][1].split("\n")
        turns = [int(l.split("|")[2]) for l in log if l.startswith("|turn|")]
        if not turns:
            continue
        t = rng.choice(turns)
        if (rid, t) not in out and human_turn(log, t):
            out.append((rid, t))
    return out


def simulate(logs, positions, seeds, dx):
    data = []
    for rid, t in positions:
        log = logs[rid][1]
        try:
            b = a.load_position(log, t, dx)
            A, B, F = a.payoff_matrix(b, "p1", "p2", seeds)
        except Exception as e:  # noqa: BLE001 - a position the loader/engine can't handle is skipped, not fatal
            print(f"skip {rid} t{t}: {e!r}")
            continue
        acts = human_turn(log.split("\n"), t)
        ia, ib = matches(b, "p1", A, acts), matches(b, "p2", B, acts)
        if not ia or not ib:
            print(f"skip {rid} t{t}: human action not in choice list {acts}")
            continue
        names = (a.name_choice(b, "p1", A[ia[0]]), a.name_choice(b, "p2", B[ib[0]]))
        data.append(dict(rid=rid, t=t, F=F, ia=ia, ib=ib, names=names, dev=(rid, t) in DEV,
                         A=[a.name_choice(b, "p1", x) for x in A], B=[a.name_choice(b, "p2", x) for x in B]))
        print(f"simulated {rid[-10:]} t{t}: {F.shape[0]}x{F.shape[1]}", flush=True)
    return data


def score(data, w, iters=3000):
    """Per position: (worse share p1, worse share p2, top pick p1, top pick p2)."""
    rows = []
    for d in data:
        M = d["F"] @ w
        p, q = a.regret_matching(M, iters)
        ev_a, ev_b = M @ q, -(p @ M)
        rows.append((worse_share(ev_a, d["ia"]), worse_share(ev_b, d["ib"]),
                     d["A"][int(np.argmax(ev_a))], d["B"][int(np.argmax(ev_b))]))
    return rows


def report(data, rows, label):
    print(f"\n{label}")
    print(f"  {'position':<16} {'side':<4} {'worse':>6}  human played  /  our top pick")
    for d, r in zip(data, rows):
        for side, k in (("p1", 0), ("p2", 1)):
            print(f"  {d['rid'][-10:]} t{d['t']:<3} {side:<4} {r[k]:6.2f}  {d['names'][k]}  /  {r[k + 2]}")
    for name, dev in (("fresh", False), ("dev", True)):
        ws = [x for d, r in zip(data, rows) if d["dev"] == dev for x in r[:2]]
        if ws:
            print(f"  {name}: mean worse {np.mean(ws):.3f}, human = our top pick {sum(x == 0 for x in ws)}/{len(ws)}, "
                  f"in top 10% {sum(x <= 0.1 for x in ws)}/{len(ws)}")


GRID = {"tailwind": [0, 0.05, 0.1, 0.15, 0.25], "boost": [0, 0.02, 0.04, 0.08, 0.12],
        "screens": [0, 0.02, 0.05, 0.1], "protect": [0, -0.05, -0.1, -0.2, -0.3, -0.5], "status": [0, 0.5, 1, 2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--cache", help="pickle of simulated matrices; created if missing (stale once features change)")
    ap.add_argument("--grid", action="store_true", help="coordinate-search the weights")
    args = ap.parse_args()

    logs = json.load(open(a.LOGS))
    if args.cache and os.path.exists(args.cache):
        data = pickle.load(open(args.cache, "rb"))
    else:
        positions = DEV + sample_positions(logs, args.n, random.Random(0))
        data = simulate(logs, positions, args.seeds, ninjask.dex(a.FORMAT))
        if args.cache:
            pickle.dump(data, open(args.cache, "wb"))

    report(data, score(data, a.WEIGHTS), f"current weights {dict(zip(a.FEATURES, a.WEIGHTS))}")
    if not args.grid:
        return

    # Coordinate search: tune on the dev positions + even-indexed fresh ones, report on the odd-indexed (held out).
    fresh = [d for d in data if not d["dev"]]
    train = [d for d in data if d["dev"]] + fresh[0::2]
    test = fresh[1::2]
    status = [a.F_[k] for k in ("slp", "frz", "par", "brn", "tox", "psn")]

    def weights(p):
        w = a.WEIGHTS.copy()
        for k in ("tailwind", "boost", "screens", "protect"):
            w[a.F_[k]] = p[k]
        w[status] = a.WEIGHTS[status] * p["status"]
        return w

    def mean_worse(ds, p):
        return np.mean([x for r in score(ds, weights(p), iters=1000) for x in r[:2]])

    cur = {k: a.WEIGHTS[a.F_[k]] for k in ("tailwind", "boost", "screens", "protect")} | {"status": 1.0}
    best = dict(cur)
    for _ in range(2):
        for k, vals in GRID.items():
            trial = {v: mean_worse(train, best | {k: v}) for v in vals}
            best[k] = min(trial, key=trial.get)
            print(f"  {k:<9} " + "  ".join(f"{v}:{x:.3f}" for v, x in trial.items()) + f"  -> {best[k]}", flush=True)
    print(f"\nCoordinate search (mean worse share; train {len(train)} positions, held-out test {len(test)})")
    for name, p in (("current", cur), ("tuned", best)):
        print(f"  {name:<8} train {mean_worse(train, p):.3f}  test {mean_worse(test, p):.3f}  {p}")

if __name__ == "__main__":
    main()
