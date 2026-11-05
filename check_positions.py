"""How well does the evaluator agree with what humans played? Also grid-searches the weights.

Usage: python check_positions.py [--n 40] [--seeds 16] [--cache FILE.pkl] [--grid] [--workers N]
                                 [--depth 2 --k 8 --seeds2 4]

Samples --n random replay positions (fixed seed) plus the 3 development positions, reads both
players' actual turn from the log, and ranks each human choice by EV against the opponent's
equilibrium mix. "worse" = share of our options ranked strictly above the human's (ties count
half): 0 = human played our top option, 0.5 = middle of the pack.
"""

import argparse
import json
import multiprocessing
import os
import pickle
import random

import numpy as np
import ninjask

import analyze as a

DEV = [("gen9championsvgc2026regmb-2635876837", 3), ("gen9championsvgc2026regmb-2635668924", 1),
       ("gen9championsvgc2026regmb-2635871725", 4)]


def human_turn(log, turn):
    """{'p1a': ('move', 'Heat Wave', 'p2b' or None, mega) | ('switch', name, species), ...}, None if a slot is unknown."""
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
            acts[slot] = ["switch", w[2].split(": ", 1)[1], w[3].split(",")[0]]
    if len(acts) != 4:
        return None
    for s in megas:
        if acts[s][0] == "move":
            acts[s].append(True)
    return acts


def matches(battle, side, choices, acts):
    """Indices of joint choices consistent with the human's logged actions for `side`."""
    team = json.loads(battle.request(side))["side"]["pokemon"]
    out = []
    for k, c in enumerate(choices):
        ok = True
        for slot, part, raw in zip("ab", a.name_choice(battle, side, c).split(" | "), c.split(", ")):
            act = acts[side + slot]
            if act[0] == "switch":
                # Name or species: the log's nickname, while the battle may know an unrevealed mon only by species
                # (EARFQUAKE = Garchomp), or by a name that isn't its forme (Sinistcha = Sinistcha-Masterpiece).
                w = raw.split()
                mon = team[int(w[1]) - 1] if w[0] == "switch" else None
                ok &= mon is not None and (mon["ident"].split(": ", 1)[1] == act[1]
                                           or mon["details"].split(",")[0] == act[2])
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


LOGS = None  # per process: set by init_worker


def init_worker():
    global LOGS
    LOGS = json.load(open(a.LOGS))


def simulate_one(job):
    """One position -> its data dict, or a skip message (str). depth2 = (k, seeds per cell) or None."""
    rid, t, seeds, depth2 = job
    log = LOGS[rid][1]
    try:
        b = a.load_position(log, t, ninjask.dex(a.FORMAT))
        A, B, F = a.payoff_matrix(b, "p1", "p2", seeds)
    except Exception as e:  # noqa: BLE001 - a position the loader/engine can't handle is skipped, not fatal
        return f"skip {rid} t{t}: {e!r}"
    acts = human_turn(log.split("\n"), t)
    ia, ib = matches(b, "p1", A, acts), matches(b, "p2", B, acts)
    if not ia or not ib:
        return f"skip {rid} t{t}: human action not in choice list {acts}"
    names = (a.name_choice(b, "p1", A[ia[0]]), a.name_choice(b, "p2", B[ib[0]]))
    d = dict(rid=rid, t=t, F=F, ia=ia, ib=ib, names=names, dev=(rid, t) in DEV,
             A=[a.name_choice(b, "p1", x) for x in A], B=[a.name_choice(b, "p2", x) for x in B])
    if depth2:
        # 2-ply values depend on the weights through each child's game: valid for a.WEIGHTS only
        d["cut_a"], d["cut_b"], d["M2"] = a.search2(b, "p1", "p2", A, B, F, *depth2)
        d["W2"] = a.WEIGHTS.copy()
    return d


def simulate(positions, seeds, workers, depth2=None):
    """Positions run in parallel (one process each, in input order)."""
    data = []
    jobs = [(rid, t, seeds, depth2) for rid, t in positions]
    with multiprocessing.Pool(workers, initializer=init_worker) as pool:
        for (rid, t), d in zip(positions, pool.imap(simulate_one, jobs)):
            if isinstance(d, str):
                print(d)
                continue
            data.append(d)
            print(f"simulated {rid[-10:]} t{t}: {d['F'].shape[0]}x{d['F'].shape[1]}", flush=True)
    return data


def score(data, w, iters=3000, depth=1):
    """Per position: (worse share p1, worse share p2, top pick p1, top pick p2).

    depth 2: the 2-ply cut ranks first, by its own EV; the other actions follow in one-turn order.
    """
    rows = []
    for d in data:
        M = d["F"] @ w
        p, q = a.regret_matching(M, iters)
        ev_a, ev_b = M @ q, -(p @ M)
        if depth == 2:
            assert np.allclose(d["W2"], w), "2-ply values were searched with other weights"
            p2, q2 = a.regret_matching(d["M2"], iters)
            ev_a[d["cut_a"]] = 1e6 + d["M2"] @ q2
            ev_b[d["cut_b"]] = 1e6 - p2 @ d["M2"]
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
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--depth", type=int, choices=(1, 2), default=1, help="2: also run the 2-ply search")
    ap.add_argument("--k", type=int, default=8, help="depth 2: actions per side kept from the one-turn ranking")
    ap.add_argument("--seeds2", type=int, default=4, help="depth 2: RNG seeds per root cell")
    args = ap.parse_args()

    logs = json.load(open(a.LOGS))
    if args.cache and os.path.exists(args.cache):
        data = pickle.load(open(args.cache, "rb"))
    else:
        positions = DEV + sample_positions(logs, args.n, random.Random(0))
        data = simulate(positions, args.seeds, args.workers, (args.k, args.seeds2) if args.depth == 2 else None)
        if args.cache:
            pickle.dump(data, open(args.cache, "wb"))

    report(data, score(data, a.WEIGHTS), f"1-ply, current weights {dict(zip(a.FEATURES, a.WEIGHTS))}")
    if all("M2" in d for d in data):
        report(data, score(data, a.WEIGHTS, depth=2), "2-ply on the top-k cut (rest in 1-ply order)")
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
