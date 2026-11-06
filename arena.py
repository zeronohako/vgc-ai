"""Bot vs bot on Reg M-C pool matchups: the yardstick for the network once a random player is beaten.

Usage: python arena.py A B [--games 200] [--workers 12] [--seeds 4] [--greedy]
  A, B: random | search | a checkpoint (runs/NAME/ckpt.pt, runs/NAME/snap_00060.pt, or runs/NAME for its ckpt.pt)

Each team pairing (two different pool teams, one battle seed) is played twice with the same teams on the same
sides and the players swapped, so a strong team or side doesn't count for either player. Prints A's score
(win 1, draw 0.5) with a 95% interval over pairings, and appends the result to runs/arena.jsonl.

Players:
- random: a uniformly random legal choice.
- a network (model.Net): sees what the player sees (closed team sheets, selfplay.SelfPlay); samples its
  policy, or takes the argmax with --greedy.
- search: analyze.py's one-turn search on the real battle, so it also knows the foe's hidden sets (stronger
  than fair; a fair CTS version needs sampled worlds). Every joint choice of both sides played with --seeds
  seeds, cells scored by the hand-written evaluator, and a choice sampled from the regret-matching equilibrium.
  Forced switches: each option pair valued by the next turn's one-turn game. Team preview: random.
  Its choice strings leave out ally-targeted moves, which the battle allows.
"""

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("RAYON_NUM_THREADS", "1")  # one engine thread per worker process

import argparse
import json
import multiprocessing
import time

import ninjask
import numpy as np
import torch
from ninjask.choices import joint_choices

import analyze as A
import model as M
import selfplay as sp
import train


class Random:
    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def choose(self, env, d):
        return (self.rng.random(d.n) * d.counts()).astype(np.int64)


class Network:
    def __init__(self, path, vocab, greedy=False):
        if os.path.isdir(path):
            path = os.path.join(path, "ckpt.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["args"]
        self.net = M.Net(vocab, d=a["d"], layers=a["layers"], heads=a["heads"])
        self.net.load_state_dict(ck["net"])
        self.net.eval()
        self.greedy = greedy

    def reseed(self, seed):
        torch.manual_seed(seed)

    def choose(self, env, d):
        return train.act(self.net, d, self.greedy)[0]


class Search:
    def __init__(self, seeds):
        self.seeds = seeds

    def reseed(self, seed):  # once per task
        self.rng = np.random.default_rng(seed)
        self.fallbacks = 0  # decisions of this task the search couldn't answer (random choice instead)

    def choose(self, env, d):
        out = (self.rng.random(d.n) * d.counts()).astype(np.int64)  # team preview stays random
        for k in np.flatnonzero(~d.preview):
            battle = env.batch.battle(int(d.battle[k]))
            me, foe = ("p1", "p2") if d.side[k] == 0 else ("p2", "p1")
            legal = {c: i for i, c in enumerate(battle.legal_choices(me))}
            try:
                choice = self.decide(battle, me, foe)
            except (ValueError, AssertionError, TypeError):
                choice = None
            if choice in legal:
                out[k] = legal[choice]
            else:
                self.fallbacks += 1
        return out

    def decide(self, battle, me, foe):
        seed0 = int(self.rng.integers(1 << 30))
        if "forceSwitch" in json.loads(battle.request(me)):
            return self.replace(battle, me, foe, seed0)
        acts, _, F = A.payoff_matrix(battle, me, foe, self.seeds, seed0)
        if not acts:
            return None
        p, _ = A.regret_matching(F @ A.WEIGHTS)
        return acts[self.rng.choice(len(acts), p=p)]

    def replace(self, battle, me, foe, seed0):
        """A switch-in (a faint or a pivot move): each of my options against each of the foe's, if it chooses
        too, valued by the position it leads to (analyze.position_value); a choice from the equilibrium."""
        opts = [joint_choices(json.loads(battle.request(s)), game_type="doubles") if battle.needs_choice(s)
                else [None] for s in (me, foe)]
        if len(opts[0]) == 1:
            return opts[0][0]
        V = np.empty((len(opts[0]), len(opts[1])))
        for i, x in enumerate(opts[0]):
            for j, y in enumerate(opts[1]):
                c = battle.copy()
                c.set_logging(True)
                c.reseed(seed0 + 7919 * i + 104729 * j)  # the real battle's RNG stays unseen
                for side, choice in ((me, x), (foe, y)):
                    if choice is not None:
                        c.choose(side, choice)
                V[i, j] = A.position_value(c, me, foe, self.seeds, seed0)
        p, _ = A.regret_matching(V)
        return opts[0][self.rng.choice(len(opts[0]), p=p)]


_W = {}  # per worker process: the two players


def make_player(spec, seeds, greedy, vocab):
    if spec == "random":
        return Random()
    if spec == "search":
        return Search(seeds)
    return Network(spec, vocab, greedy)


def _init(a, b, seeds, greedy):
    torch.set_num_threads(1)
    vocab = ninjask.dex(sp.FORMAT).vocab()
    _W["teams"] = sp.load_pool()
    _W["players"] = [make_player(s, seeds, greedy, vocab) for s in (a, b)]


def play(task):
    """Play a list of games (p1 team, p2 team, battle seed, (pairing, side of A)) in one env; returns
    [(pairing, side of A, winner side or None, turns)] and each player's (seconds, decisions)."""
    games, seed = task
    players = _W["players"]
    for k, p in enumerate(players):
        p.reseed(seed * 2 + k)
    env = sp.SelfPlay(_W["teams"], len(games), matchups=games)
    a_side = np.array([g[3][1] for g in games])  # by battle slot: slot i plays game i, and only that one
    rng = np.random.default_rng(seed)
    refused = {}  # (battle, side) -> choice indices the engine refused at this decision
    results, timing = [], np.zeros((2, 2))
    while env.active.any():
        d = env.observe()
        choice = np.zeros(d.n, np.int64)
        mine = d.side == a_side[d.battle]
        for k, (player, mask) in enumerate(zip(players, (mine, ~mine))):
            if mask.any():
                idx = np.flatnonzero(mask)
                t = time.perf_counter()
                choice[idx] = player.choose(env, train.subset(d, idx))
                timing[k] += (time.perf_counter() - t, len(idx))
        for k in range(d.n):  # never resubmit a refused choice (a greedy player would loop)
            bad = refused.get((int(d.battle[k]), int(d.side[k])))
            if bad and choice[k] in bad:
                ok = [i for i in range(d.counts()[k]) if i not in bad]
                choice[k] = rng.choice(ok)
        ended = env.step(choice)
        for k in range(d.n):
            key = (int(d.battle[k]), int(d.side[k]))
            if env.refused[k]:
                refused.setdefault(key, set()).add(int(choice[k]))
            else:
                refused.pop(key, None)
        for (slot, winner, turns), tag in zip(ended, env.finished):
            results.append((tag[0], tag[1], winner, turns))
            a_side[slot] = -1
            for side in (0, 1):
                refused.pop((slot, side), None)
    return results, timing, sum(getattr(p, "fallbacks", 0) for p in players)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=200, help="rounded up to an even number (pairings x 2)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, default=4, help="search: RNG seeds per cell")
    ap.add_argument("--greedy", action="store_true", help="networks take their policy's argmax")
    ap.add_argument("--min-rating", type=int, default=0, help="only pool teams of players rated this or more")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=os.path.join("runs", "arena.jsonl"))
    args = ap.parse_args()

    teams = sp.load_pool(min_rating=args.min_rating, meta_path="data/teams_regmc_meta.json")
    full = sp.load_pool()
    index = {t: k for k, t in enumerate(full)}  # workers index the full pool
    rng = np.random.default_rng(args.seed)
    pairings = (args.games + 1) // 2
    tasks = []
    for j in range(pairings):
        x, y = (index[teams[i]] for i in rng.choice(len(teams), 2, replace=False))
        s = int(rng.integers(1 << 31))
        tasks.append(([(x, y, s, (j, 0)), (x, y, s, (j, 1))], args.seed * 100003 + j))

    t0 = time.time()
    results, timing, fallbacks = [], np.zeros((2, 2)), 0
    init = (args.a, args.b, args.seeds, args.greedy)
    if args.workers <= 1:
        _init(*init)
        outs = map(play, tasks)
    else:
        pool = multiprocessing.Pool(args.workers, initializer=_init, initargs=init)
        outs = pool.imap_unordered(play, tasks)
    for done, (res, tim, fb) in enumerate(outs, 1):
        results += res
        timing += tim
        fallbacks += fb
        if done % max(1, pairings // 10) == 0:
            sc = np.mean([0.5 if w is None else float(w == s) for _, s, w, _ in results])
            print(f"  {done}/{pairings} pairings, {time.time() - t0:.0f} s: A {sc:.1%}", flush=True)

    score = {}
    for j, s, w, _ in results:
        score.setdefault(j, []).append(0.5 if w is None else float(w == s))
    pair = np.array([sum(v) for v in score.values()])  # A's points per pairing, 0..2
    wr = pair.mean() / 2
    ci = 1.96 * pair.std(ddof=1) / 2 / np.sqrt(len(pair)) if len(pair) > 1 else float("nan")
    by_side = [np.mean([0.5 if w is None else float(w == s) for _, s2, w, _ in results if s2 == s]) for s in (0, 1)]
    turns = [t for *_, t in results]
    draws = sum(w is None for _, _, w, _ in results)
    per = [1e3 * t / max(n, 1) for t, n in timing]
    print(f"A = {args.a}, B = {args.b}: {len(results)} games ({len(pair)} pairings), {time.time() - t0:.0f} s")
    print(f"A scores {wr:.1%} +- {ci:.1%} (95%); as p1 {by_side[0]:.1%}, as p2 {by_side[1]:.1%}; draws {draws}")
    print(f"pairings won both / split / lost both: {(pair == 2).sum()} / {((pair > 0) & (pair < 2)).sum()} / "
          f"{(pair == 0).sum()}; turns median {int(np.median(turns))}")
    print(f"ms per decision: A {per[0]:.1f}, B {per[1]:.1f}" + (f"; search fallbacks {fallbacks}" if fallbacks else ""))
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps({"a": args.a, "b": args.b, "games": len(results), "score_a": round(wr, 4),
                            "ci95": round(ci, 4), "as_p1": round(by_side[0], 4), "as_p2": round(by_side[1], 4),
                            "draws": draws, "seeds": args.seeds, "greedy": args.greedy, "min_rating": args.min_rating,
                            "seed": args.seed, "seconds": round(time.time() - t0), "ms_per_decision": per,
                            "date": time.strftime("%Y-%m-%d %H:%M")}) + "\n")


if __name__ == "__main__":
    main()
