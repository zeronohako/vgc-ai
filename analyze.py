"""Best moves for one VGC position, rebuilt from a Showdown OTS replay log.

Usage: python analyze.py <replay_id> <turn> [--side p1|p2] [--seeds 16] [--depth 1|2] [--k 8] [--seeds2 4]

Rebuilds the position at the start of <turn> in Ninjask, lists every joint
action for both sides, plays each pair with several RNG seeds, scores the
result with a heuristic, and solves the payoff matrix with regret matching.
--depth 2 re-values the top-k actions per side by the one-turn game of the
position each pair leads to (2-ply).
"""

import argparse
import json
import multiprocessing
import os
import time

import numpy as np
import ninjask
from ninjask.choices import joint_choices
from scipy.optimize import linprog

FORMAT = "gen9championsvgc2026regmb"  # the replay dataset covers Reg M-A/M-B only
LOGS = "data/logs_regmb.json"


def brought(log, side):
    """Species that side sent out at any point in the whole game."""
    return {l.split("|")[3].split(",")[0] for l in log if l.startswith(f"|switch|{side}")}


def world_of_foe(dx, lines, me, foe_brought):
    """The battle as `me` sees it, keeping only the foe side (filled with default spreads).

    Own side is unusable from a spectator log (no requests: HP in %, no stats),
    so each side is taken from the other player's view instead.
    """
    t = ninjask.Tracker(dx, me)
    t.feed(lines)
    view = json.loads(t.view())
    foe_view = view["sides"][1 if me == "p1" else 0]["pokemon"]
    idx = [k for k, m in enumerate(foe_view) if m["preview"] in foe_brought]
    # ponytail: assumes foe spreads = Ninjask's default guess (32 atk/spe-ish); real spreads unknown in OTS
    assumptions = json.dumps({"brought": idx}) if len(idx) == 4 else None
    return json.loads(t.world(assumptions).projection())


def stall_streaks(log, turn):
    """{'p1a: Blaziken': n} = consecutive turns (ending at turn-1) that mon landed a Protect-like move.

    The tracker projection drops the `stall` volatile, so without this a repeat Protect always works.
    """
    turns = [i for i, l in enumerate(log) if l.startswith("|turn|")][:turn]  # starts of turns 0..turn-1
    streak, alive = {}, None
    for a, b in reversed(list(zip(turns, turns[1:]))):  # turn-1, turn-2, ...
        hit = {l.split("|")[2] for l in log[a:b] if l.startswith("|-singleturn|")
               and any(p in l for p in ("Protect", "Endure", "Shield", "Bunker", "Silk Trap", "Bulwark", "Obstruct"))}
        alive = hit if alive is None else alive & hit
        if not alive:
            break
        for ident in alive:
            streak[ident] = streak.get(ident, 0) + 1
    return streak


def load_position(log_text, turn, dx):
    log = log_text.split("\n")
    cut = log.index(f"|turn|{turn}")
    lines = log[: cut + 1]
    p1 = world_of_foe(dx, lines, "p2", brought(log, "p1"))
    p2 = world_of_foe(dx, lines, "p1", brought(log, "p2"))
    p2["sides"] = [p1["sides"][0], p2["sides"][1]]
    for ident, n in stall_streaks(log, turn).items():
        side, name = ident[:2], ident.split(": ", 1)[1]
        for m in p2["sides"][0 if side == "p1" else 1]["pokemon"]:
            if m["isActive"] and name in (m.get("name"), m["species"]):
                m["volatiles"]["stall"] = {"counter": 3**n, "duration": 1}
                m["volatileOrder"].append("stall")
    b = ninjask.Battle.from_projection(dx, json.dumps(p2), 0)
    b.set_logging(False)
    return b


# The score is FEATURES . WEIGHTS (my side minus foe), so weights can be re-tuned without re-simulating.
# ponytail: hand-set weights (see check_positions.py); a value net replaces this later.
#   hp: per mon 0.5 alive + 0.5 x HP fraction; tailwind: turns left; boost: stat stages, all stats equal;
#   screens: Reflect + Light Screen turns left (Aurora Veil counts as both);
#   protect: Protect-type moves chosen this turn (tempo cost the one-turn horizon can't see);
#   status: mons with that status. Ignores Trick Room.
FEATURES = ("hp", "tailwind", "boost", "screens", "protect", "slp", "frz", "par", "brn", "tox", "psn")
WEIGHTS = np.array([1.0, 0.15, 0.08, 0.0, 0.0, -0.3, -0.3, -0.15, -0.15, -0.15, -0.1])
F_ = {name: k for k, name in enumerate(FEATURES)}
WIN = 10.0  # hp-feature value of a won game


STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
SIDE_CONDS = ("tailwind", "reflect", "lightscreen", "auroraveil")
PROTECTS = {"protect", "detect", "spikyshield", "banefulbunker", "kingsshield", "silktrap", "burningbulwark", "obstruct"}


def snapshot(battle):
    """(conds, boosts, mons) from one projection() (~0.4 ms: once per root, never per cell).

    conds = {'p1': {'tailwind': turns left, ...}, 'p2': ...}; boosts = {'p1a': {'atk': 1, ...}} for active mons;
    mons = {('p1', name): (hp, maxhp, status)} for every brought mon.
    """
    conds, boosts, mons = {}, {}, {}
    for s in json.loads(battle.projection())["sides"]:
        conds[s["id"]] = {c: s["sideConditions"].get(c, {}).get("duration", 0) for c in SIDE_CONDS}
        for m in s["pokemon"]:
            if m["isActive"] and m["hp"] > 0:
                boosts[s["id"] + "ab"[m["position"]]] = dict(m["boosts"])
            mons[(s["id"], m["name"])] = (m["hp"], m["maxhp"], m["status"] or None)
    return conds, boosts, mons


def side_conditions(battle):
    """{'p1': {'tailwind': turns left, ...}, 'p2': {...}}. Needs projection() (~0.4 ms): avoid per cell."""
    return snapshot(battle)[0]


def active_boosts(battle):
    """{'p1a': {'atk': 1, ...}, ...} for the active mons. Uses projection() (slow): call once per root."""
    return snapshot(battle)[1]


def hp_field(field):
    """'96/157 par' -> (96, 'par'); '0 fnt' -> (0, None). The omniscient log has exact HP."""
    w = field.split()
    if w[0] == "0" or w[-1] == "fnt":
        return 0, None
    return int(w[0].split("/")[0]), (w[1] if len(w) > 1 else None)


# Log lines each parser reads; a C-level startswith skips the rest (most of a turn's log) before any split.
HP_LINES = ("|-damage|", "|-heal|", "|-sethp|", "|switch|", "|drag|", "|faint|", "|-status|", "|-curestatus|",
            "|-cureteam|")
BOOST_LINES = ("|-boost|", "|-unboost|", "|-setboost|", "|-clearnegativeboost|", "|-clearboost|",
               "|-clearallboost", "|switch|", "|drag|", "|faint|")
COND_LINES = ("|-sidestart|", "|-sideend|", "|-swapsideconditions")


def apply_hp_log(mons, lines):
    """{('p1', name): (hp, status)} after one turn: the root's `mons` (from snapshot) plus this turn's log.

    Every HP field carries the status too, so damage/heal/switch lines resync both. Even "silent" changes
    are logged (Regenerator: `-heal ... [silent]`). Checked against requests in test_analyze.
    """
    m = {k: [v[0], v[2]] for k, v in mons.items()}
    for l in lines:
        if not l.startswith(HP_LINES):
            continue
        w = l.split("|")
        if len(w) < 4 or ": " not in w[2]:
            continue
        kind, key = w[1], (w[2][:2], w[2].split(": ", 1)[1])
        if key not in m:
            continue
        if kind in ("-damage", "-heal", "-sethp"):
            m[key] = list(hp_field(w[3]))
        elif kind in ("switch", "drag"):
            m[key] = list(hp_field(w[4]))
        elif kind == "faint":
            m[key][0] = 0
        elif kind == "-status":
            m[key][1] = w[3]
        elif kind == "-curestatus":
            m[key][1] = None
        elif kind == "-cureteam":
            for k in m:
                if k[0] == key[0]:
                    m[k][1] = None
    return {k: tuple(v) for k, v in m.items()}


def team_features(hp, maxhp, side):
    """FEATURES vector for one side's hp and status (hp = apply_hp_log output, maxhp = {key: max hp})."""
    f = np.zeros(len(FEATURES))
    for key, (cur, status) in hp.items():
        if key[0] != side or cur == 0:
            continue
        f[0] += 0.5 + 0.5 * cur / maxhp[key]
        if status in F_:
            f[F_[status]] += 1
    return f


def apply_boost_log(boosts, lines):
    """Boosts after one turn: the root's `boosts` plus this turn's log messages."""
    # ponytail: skips -copyboost/-swapboost/-invertboost (2 of ~2300 boost lines in the dataset)
    b = {k: dict(v) for k, v in boosts.items()}
    for l in lines:
        if not l.startswith(BOOST_LINES):
            continue
        if l.startswith("|-clearallboost"):  # Haze: in log order, so later boosts this turn still count
            b = {k: dict.fromkeys(STATS, 0) for k in b}
            continue
        w = l.split("|")
        if len(w) < 3:
            continue
        kind, slot = w[1], w[2][:3]
        if kind in ("switch", "drag"):
            b[slot] = dict.fromkeys(STATS, 0)
        elif kind == "faint":
            b.pop(slot, None)
        elif slot not in b:
            continue
        elif kind in ("-boost", "-unboost"):
            n = int(w[4]) * (1 if kind == "-boost" else -1)
            b[slot][w[3]] = max(-6, min(6, b[slot][w[3]] + n))
        elif kind == "-setboost":
            b[slot][w[3]] = int(w[4])
        elif kind == "-clearnegativeboost":
            b[slot] = {k: max(v, 0) for k, v in b[slot].items()}
        elif kind == "-clearboost":
            b[slot] = dict.fromkeys(STATS, 0)
    return b


def boost_stages(boosts, side):
    return sum(sum(v.values()) for k, v in boosts.items() if k.startswith(side))


def chosen_moves(req, choice):
    """Move ids this joint choice ('move 2 1, switch 3') picks."""
    return [req["active"][slot]["moves"][int(w[1]) - 1]["id"]
            for slot, w in enumerate(p.split() for p in choice.split(", ")) if w[0] == "move"]


FRESH = {"tailwind": 3, "reflect": 4, "lightscreen": 4, "auroraveil": 4}  # turns left after the turn it's set


def turn_conds(battle, lines, aged, clay=frozenset()):
    """Side conditions after this turn. Ones already up just count down (re-setting fails), so start from
    `aged` (the root's, minus one turn) and apply what the log shows set (FRESH) or ended (0). Screens set by
    a side holding Light Clay (`clay`: sides) and Court Change's swap read projection() instead."""
    out = aged
    for l in lines:
        if not l.startswith(COND_LINES):
            continue
        w = l.split("|")
        if w[1] == "-swapsideconditions":
            return side_conditions(battle)
        side, cond = w[2][:2], w[3].removeprefix("move: ").lower().replace(" ", "")
        if cond not in FRESH:
            continue  # hazards etc.: not scored
        if w[1] == "-sidestart" and cond != "tailwind" and side in clay:
            return side_conditions(battle)
        if out is aged:
            out = {s: dict(c) for s, c in aged.items()}
        out[side][cond] = FRESH[cond] if w[1] == "-sidestart" else 0
    return out


def features(battle, me, foe, conds, boosts, hp, maxhp):
    """FEATURES vector, my side minus foe (except `protect`, which payoff_matrix fills). Score = features(...) @ WEIGHTS.

    conds = side conditions after this turn (turn_conds). boosts and hp are this turn's results rebuilt from
    its log (apply_boost_log, apply_hp_log); maxhp = {key: max hp}.
    """
    f = np.zeros(len(FEATURES))
    if battle.ended:
        f[0] = {me: WIN, foe: -WIN}.get(battle.winner_side, 0.0)
        return f
    screens = {s: c["reflect"] + c["lightscreen"] + 2 * c["auroraveil"] for s, c in conds.items()}
    f += team_features(hp, maxhp, me) - team_features(hp, maxhp, foe)
    f[F_["tailwind"]] = conds[me]["tailwind"] - conds[foe]["tailwind"]
    f[F_["screens"]] = screens[me] - screens[foe]
    f[F_["boost"]] = boost_stages(boosts, me) - boost_stages(boosts, foe)
    return f


def finish_turn(battle, start):
    """U-turn/Volt Switch/Parting Shot pause the turn for a switch-in; answer until the turn's upkeep runs.

    Without this, every pivot cell is scored halfway through the turn (slower mons haven't moved yet).
    """
    # ponytail: picks the first legal switch-in; the best one needs a nested search
    while not battle.ended and not any(l.startswith("|upkeep") for l in battle.log("omniscient", start)):
        side = next((s for s in ("p1", "p2") if battle.needs_choice(s)), None)
        if side is None:
            return
        battle.choose(side, joint_choices(json.loads(battle.request(side)), game_type="doubles")[0])


def payoff_matrix(battle, me, foe, seeds, seed0=0):
    """Returns (A, B, F): F[i, j] = mean FEATURES vector of my A[i] vs their B[j] over RNG seeds
    seed0 .. seed0+seeds-1. Payoffs = F @ WEIGHTS."""
    ra, rb = json.loads(battle.request(me)), json.loads(battle.request(foe))
    A = joint_choices(ra, game_type="doubles")
    B = joint_choices(rb, game_type="doubles")
    root, root_boosts, root_mons = snapshot(battle)
    maxhp = {k: v[1] for k, v in root_mons.items()}
    aged = {s: {c: max(d - 1, 0) for c, d in cs.items()} for s, cs in root.items()}
    clay = {s for s, r in ((me, ra), (foe, rb)) if any(m["item"] == "lightclay" for m in r["side"]["pokemon"])}
    prot_a = [sum(m in PROTECTS for m in chosen_moves(ra, a)) for a in A]
    prot_b = [sum(m in PROTECTS for m in chosen_moves(rb, b)) for b in B]
    F = np.full((len(A), len(B), len(FEATURES)), np.nan)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            total = np.zeros(len(FEATURES))
            for s in range(seed0, seed0 + seeds):
                c = battle.copy()
                c.set_logging(True)  # +~13 us; HP, status and boosts come from the turn's log (requests: ~140 us)
                c.reseed(s)
                start = c.log_len()
                try:
                    c.choose(me, a)
                    c.choose(foe, b)
                except ValueError:
                    break  # engine rejects the pair: drop the cell
                finish_turn(c, start)
                lines = c.log("omniscient", start)
                total += features(c, me, foe, turn_conds(c, lines, aged, clay), apply_boost_log(root_boosts, lines),
                                  apply_hp_log(root_mons, lines), maxhp)
            else:
                F[i, j] = total / seeds
                F[i, j, F_["protect"]] = prot_a[i] - prot_b[j]
    # An illegal action (e.g. switching while trapped by Shadow Tag) is rejected in its whole row/column:
    # drop those first, or every opposing action touching them gets dropped too.
    bad = np.isnan(F).any(axis=2)
    rows = ~bad.all(axis=1)
    cols = ~bad[rows].all(axis=0)
    rows &= ~(bad[:, cols].any(axis=1))  # leftover mixed rejections: drop the row, as before
    return [a for a, k in zip(A, rows) if k], [b for b, k in zip(B, cols) if k], F[rows][:, cols]


def regret_matching(M, iters=5000):
    """Approximate Nash of the zero-sum matrix game; returns (row mix, column mix)."""
    n, m = M.shape
    rp, rq = np.zeros(n), np.zeros(m)
    sp, sq = np.zeros(n), np.zeros(m)
    for _ in range(iters):
        p = np.maximum(rp, 0)
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1 / n)
        q = np.maximum(rq, 0)
        q = q / q.sum() if q.sum() > 0 else np.full(m, 1 / m)
        u_rows, u_cols = M @ q, -(p @ M)
        rp += u_rows - p @ u_rows
        rq += u_cols - q @ u_cols
        sp += p
        sq += q
    return sp / sp.sum(), sq / sq.sum()


def game_value(M):
    """Value of the zero-sum matrix game M for the row (max) player: the saddle point if any, else by LP (~5 ms
    for 60x55, vs ~140 ms for regret_matching)."""
    lo, hi = M.min(axis=1).max(), M.max(axis=0).min()
    if hi - lo < 1e-12:
        return float(lo)
    n, m = M.shape
    res = linprog(np.r_[np.zeros(n), -1.0], A_ub=np.c_[-M.T, np.ones(m)], b_ub=np.zeros(m),
                  A_eq=np.r_[np.ones(n), 0.0][None], b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)],
                  method="highs")
    assert res.success, res.message
    return float(-res.fun)


# 2-ply search: each cell of the root matrix is valued by the one-turn game of the position it leads to.
# ponytail: depth 2 only, and the child's leaves still use the one-turn heuristic.

def position_value(battle, me, foe, seeds, seed0=0):
    """Value for `me` of a position between turns: its own one-turn matrix game (heuristic leaves), by LP.

    Fainted mons are replaced first. Both sides pick at once, so that is a small matrix game of its own.
    """
    if battle.ended:
        return WIN * WEIGHTS[F_["hp"]] * {me: 1, foe: -1}.get(battle.winner_side, 0)
    reqs = {s: json.loads(battle.request(s)) for s in (me, foe) if battle.needs_choice(s)}
    if not any("forceSwitch" in r for r in reqs.values()):
        return game_value(payoff_matrix(battle, me, foe, seeds, seed0)[2] @ WEIGHTS)
    opts = [joint_choices(reqs[s], game_type="doubles") if s in reqs else [None] for s in (me, foe)]
    V = np.empty((len(opts[0]), len(opts[1])))
    for i, x in enumerate(opts[0]):
        for j, y in enumerate(opts[1]):
            c = battle.copy()
            for side, choice in ((me, x), (foe, y)):
                if choice is not None:
                    c.choose(side, choice)
            V[i, j] = position_value(c, me, foe, seeds, seed0)
    return game_value(V)


def child_value(battle, me, foe, a, b, seed, child_seeds):
    """position_value after my `a` vs their `b` with RNG `seed`; NaN if the engine rejects the pair."""
    c = battle.copy()
    c.set_logging(True)  # finish_turn reads the log
    c.reseed(seed)
    start = c.log_len()
    try:
        c.choose(me, a)
        c.choose(foe, b)
    except ValueError:
        return np.nan
    finish_turn(c, start)
    # the child's own seeds differ from the root turn's, so one lucky seed doesn't repeat in both turns
    return position_value(c, me, foe, child_seeds, seed0=1000 * (seed + 1))


_WORKER = {}  # per process: the root battle and search settings (set by _init_worker)


def _init_worker(format_id, projection, me, foe, child_seeds):
    b = ninjask.Battle.from_projection(ninjask.dex(format_id), projection, 0)  # exact round trip (test_analyze)
    b.set_logging(False)
    _WORKER.update(battle=b, me=me, foe=foe, child_seeds=child_seeds)


def _child_job(job):
    i, j, a, b, s = job
    w = _WORKER
    return i, j, child_value(w["battle"], w["me"], w["foe"], a, b, s, w["child_seeds"])


def deepen(battle, me, foe, A, B, seeds, child_seeds=1, workers=1):
    """2-ply matrix: M2[i, j] = mean over root seeds 0..seeds-1 of child_value(A[i], B[j]); NaN if rejected.

    Each child is a full one-turn matrix with `child_seeds` seeds (~0.1-0.2 s), so keep A and B to a top-k cut.
    workers > 1 spreads the children over processes (each rebuilds the root from its projection).
    """
    jobs = [(i, j, a, b, s) for i, a in enumerate(A) for j, b in enumerate(B) for s in range(seeds)]
    total = np.zeros((len(A), len(B)))
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        init = (FORMAT, battle.projection(), me, foe, child_seeds)
        with ctx.Pool(workers, initializer=_init_worker, initargs=init) as pool:
            results = list(pool.imap_unordered(_child_job, jobs, chunksize=2))
    else:
        results = [(i, j, child_value(battle, me, foe, a, b, s, child_seeds)) for i, j, a, b, s in jobs]
    for i, j, v in results:
        total[i, j] += v
    return total / seeds


def top_k(ev, mix, k):
    """Indices of the k best actions by EV, plus any the equilibrium plays at least 1% (in EV order)."""
    keep = set(np.argsort(-ev)[:k]) | set(np.flatnonzero(mix >= 0.01))
    return sorted(keep, key=lambda i: -ev[i])


def search2(battle, me, foe, A, B, F, k, seeds, child_seeds=1, workers=1):
    """2-ply on a top-k cut of the one-turn matrix (A, B, F from payoff_matrix).

    Returns (ia, ib, M2): my A[ia[x]] vs their B[ib[y]] is worth M2[x, y]. Rows/columns the engine rejected
    are dropped.
    """
    M = F @ WEIGHTS
    p, q = regret_matching(M)
    ia, ib = top_k(M @ q, p, k), top_k(-(p @ M), q, k)
    M2 = deepen(battle, me, foe, [A[i] for i in ia], [B[j] for j in ib], seeds, child_seeds, workers)
    rows, cols = ~np.isnan(M2).all(axis=1), ~np.isnan(M2).all(axis=0)
    rows &= ~np.isnan(M2[:, cols]).any(axis=1)
    return [i for i, r in zip(ia, rows) if r], [j for j, c in zip(ib, cols) if c], M2[rows][:, cols]


def name_choice(battle, side, choice):
    """'move 1 2, switch 3' -> 'Iron Head -> p2b, switch Garchomp'."""
    req = json.loads(battle.request(side))
    team = req["side"]["pokemon"]
    out = []
    for slot, part in enumerate(choice.split(", ")):
        w = part.split()
        if w[0] == "move":
            s = req["active"][slot]["moves"][int(w[1]) - 1]["move"]
            if len(w) > 2 and w[2] != "mega":
                t = int(w[2])
                foe = "p2" if side == "p1" else "p1"
                s += f" -> {foe if t > 0 else side}{'ab'[abs(t) - 1]}"
            if "mega" in w:
                s += " (mega)"
        elif w[0] == "switch":
            s = "switch " + team[int(w[1]) - 1]["ident"].split(": ")[1]
        else:
            s = part
        out.append(s)
    return " | ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("turn", type=int)
    ap.add_argument("--side", default="p1")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--depth", type=int, choices=(1, 2), default=1)
    ap.add_argument("--k", type=int, default=8, help="depth 2: actions per side kept from the one-turn ranking")
    ap.add_argument("--seeds2", type=int, default=4, help="depth 2: RNG seeds per root cell (one child each)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    me, foe = args.side, "p2" if args.side == "p1" else "p1"

    dx = ninjask.dex(FORMAT)
    logs = json.load(open(LOGS))
    battle = load_position(logs[args.replay][1], args.turn, dx)

    proj = json.loads(battle.projection())
    for s in proj["sides"]:
        mons = ", ".join(f"{m['species']} {m['hp']}/{m['maxhp']}" for m in s["pokemon"] if m["isActive"])
        print(f"{s['id']} active: {mons}")

    t0 = time.time()
    A, B, F = payoff_matrix(battle, me, foe, args.seeds)
    M = F @ WEIGHTS
    p, q = regret_matching(M)
    ev = M @ q  # each of our actions vs the opponent's equilibrium mix
    print(f"\n{len(A)}x{len(B)} matrix, {args.seeds} seeds, {time.time() - t0:.1f}s, game value {p @ M @ q:+.2f}\n")

    if args.depth == 1:
        print(f"Best for {me} (EV vs opponent's mix, equilibrium probability):")
        for i in np.argsort(-ev)[: args.top]:
            print(f"  {ev[i]:+.2f}  {p[i]:5.1%}  {name_choice(battle, me, A[i])}")
        print(f"\nOpponent's likely plays:")
        for j in np.argsort(-q)[: args.top]:
            if q[j] >= 0.01:
                print(f"  {q[j]:5.1%}  {name_choice(battle, foe, B[j])}")
        return

    t0 = time.time()
    ia, ib, M2 = search2(battle, me, foe, A, B, F, args.k, args.seeds2, workers=args.workers)
    p2, q2 = regret_matching(M2)
    ev2 = M2 @ q2
    print(f"2-ply on the top {len(ia)}x{len(ib)}, {args.seeds2} seeds per cell, {time.time() - t0:.1f}s, "
          f"game value {p2 @ M2 @ q2:+.2f}\n")
    print(f"Best for {me} (2-ply EV vs opponent's mix, probability; 1-ply EV):")
    for x in np.argsort(-ev2)[: args.top]:
        print(f"  {ev2[x]:+.2f}  {p2[x]:5.1%}  ({ev[ia[x]]:+.2f})  {name_choice(battle, me, A[ia[x]])}")
    print(f"\nOpponent's likely plays (2-ply probability; 1-ply):")
    for y in np.argsort(-q2)[: args.top]:
        if q2[y] >= 0.01:
            print(f"  {q2[y]:5.1%}  ({q[ib[y]]:5.1%})  {name_choice(battle, foe, B[ib[y]])}")


if __name__ == "__main__":
    main()
