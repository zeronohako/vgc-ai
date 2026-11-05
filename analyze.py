"""Best moves for one VGC position, rebuilt from a Showdown OTS replay log.

Usage: python analyze.py <replay_id> <turn> [--side p1|p2] [--seeds 16]

Rebuilds the position at the start of <turn> in Ninjask, lists every joint
action for both sides, plays each pair with several RNG seeds, scores the
result with a heuristic, and solves the payoff matrix with regret matching.
"""

import argparse
import json
import time

import numpy as np
import ninjask
from ninjask.choices import joint_choices

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


def team_features(battle, side):
    """FEATURES vector for one side, minus tailwind and boost (those come from the whole battle)."""
    req = battle.request(side)
    mons = json.loads(req)["side"]["pokemon"] if req else None
    if mons is None:
        side_idx = 0 if side == "p1" else 1
        mons = [{"condition": f"{m['hp']}/{m['maxhp']}"} for m in json.loads(battle.projection())["sides"][side_idx]["pokemon"]]
    f = np.zeros(len(FEATURES))
    for m in mons:
        hp = m["condition"].split()[0]
        if hp == "0" or m["condition"].endswith("fnt"):
            continue
        cur, mx = map(int, hp.split("/"))
        f[0] += 0.5 + 0.5 * cur / mx
        status = m["condition"].split()[1] if " " in m["condition"] else None
        if status in FEATURES:
            f[FEATURES.index(status)] += 1
    return f


STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def active_boosts(battle):
    """{'p1a': {'atk': 1, ...}, ...} for the active mons. Uses projection() (slow): call once per root."""
    out = {}
    for s in json.loads(battle.projection())["sides"]:
        for m in s["pokemon"]:
            if m["isActive"] and m["hp"] > 0:
                out[s["id"] + "ab"[m["position"]]] = dict(m["boosts"])
    return out


def apply_boost_log(boosts, lines):
    """Boosts after one turn: the root's `boosts` plus this turn's log messages."""
    # ponytail: skips -copyboost/-swapboost/-invertboost (2 of ~2300 boost lines in the dataset)
    b = {k: dict(v) for k, v in boosts.items()}
    for l in lines:
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
    if any(l.startswith("|-clearallboost") for l in lines):
        b = {k: dict.fromkeys(STATS, 0) for k in b}
    return b


def boost_stages(boosts, side):
    return sum(sum(v.values()) for k, v in boosts.items() if k.startswith(side))


SIDE_CONDS = ("tailwind", "reflect", "lightscreen", "auroraveil")  # each is also the id of the move that sets it
SCREEN_REMOVERS = {"brickbreak", "psychicfangs", "ragingbull", "defog", "courtchange"}
PROTECTS = {"protect", "detect", "spikyshield", "banefulbunker", "kingsshield", "silktrap", "burningbulwark", "obstruct"}


def side_conditions(battle):
    """{'p1': {'tailwind': turns left, ...}, 'p2': {...}}. Needs projection() (~0.9 ms): avoid per cell."""
    return {s["id"]: {c: s["sideConditions"].get(c, {}).get("duration", 0) for c in SIDE_CONDS}
            for s in json.loads(battle.projection())["sides"]}


def chosen_moves(req, choice):
    """Move ids this joint choice ('move 2 1, switch 3') picks."""
    return [req["active"][slot]["moves"][int(w[1]) - 1]["id"]
            for slot, w in enumerate(p.split() for p in choice.split(", ")) if w[0] == "move"]


def changes_conditions(moves, up):
    """Can these moves change side conditions other than by counting down? (`up` = this side's root conditions)"""
    return any(m in SCREEN_REMOVERS or (m in up and not up[m]) for m in moves)


def features(battle, me, foe, conds=None, boosts=None):
    """FEATURES vector, my side minus foe (except `protect`, which payoff_matrix fills). Score = features(...) @ WEIGHTS.
    conds = side conditions after this turn if known; None reads them from the battle (slow).
    boosts = active boosts after this turn (from apply_boost_log); None reads them from the battle (slow)."""
    f = np.zeros(len(FEATURES))
    if battle.ended:
        f[0] = {me: WIN, foe: -WIN}.get(battle.winner_side, 0.0)
        return f
    cd = conds or side_conditions(battle)
    boosts = active_boosts(battle) if boosts is None else boosts
    screens = {s: c["reflect"] + c["lightscreen"] + 2 * c["auroraveil"] for s, c in cd.items()}
    f += team_features(battle, me) - team_features(battle, foe)
    f[F_["tailwind"]] = cd[me]["tailwind"] - cd[foe]["tailwind"]
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


def payoff_matrix(battle, me, foe, seeds):
    """Returns (A, B, F): F[i, j] = mean FEATURES vector of my A[i] vs their B[j]. Payoffs = F @ WEIGHTS."""
    ra, rb = json.loads(battle.request(me)), json.loads(battle.request(foe))
    A = joint_choices(ra, game_type="doubles")
    B = joint_choices(rb, game_type="doubles")
    # Side conditions already up just count down (re-setting fails), so only cells that set a fresh one
    # or can remove screens need projection().
    root = side_conditions(battle)
    aged = {s: {c: max(d - 1, 0) for c, d in cs.items()} for s, cs in root.items()}
    mv_a, mv_b = [chosen_moves(ra, a) for a in A], [chosen_moves(rb, b) for b in B]
    fresh_a = [changes_conditions(m, root[me]) for m in mv_a]
    fresh_b = [changes_conditions(m, root[foe]) for m in mv_b]
    prot_a = [sum(m in PROTECTS for m in ms) for ms in mv_a]
    prot_b = [sum(m in PROTECTS for m in ms) for ms in mv_b]
    root_boosts = active_boosts(battle)
    F = np.full((len(A), len(B), len(FEATURES)), np.nan)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            conds = None if fresh_a[i] or fresh_b[j] else aged
            total = np.zeros(len(FEATURES))
            for s in range(seeds):
                c = battle.copy()
                c.set_logging(True)  # +~17 us; the turn's log carries boost changes (projection() would be ~900 us)
                c.reseed(s)
                start = c.log_len()
                try:
                    c.choose(me, a)
                    c.choose(foe, b)
                except ValueError:
                    break  # engine rejects the pair: drop the cell
                finish_turn(c, start)
                total += features(c, me, foe, conds, apply_boost_log(root_boosts, c.log("omniscient", start)))
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

    print(f"Best for {me} (EV vs opponent's mix, equilibrium probability):")
    for i in np.argsort(-ev)[: args.top]:
        print(f"  {ev[i]:+.2f}  {p[i]:5.1%}  {name_choice(battle, me, A[i])}")
    print(f"\nOpponent's likely plays:")
    for j in np.argsort(-q)[: args.top]:
        if q[j] >= 0.01:
            print(f"  {q[j]:5.1%}  {name_choice(battle, foe, B[j])}")


if __name__ == "__main__":
    main()
