"""Self-play environment: many Reg M-C battles stepped together by Ninjask 0.4's `Batch`.

Usage: python selfplay.py [--n 256] [--secs 20] [--ots]   # random-policy benchmark on the team pool

Closed team sheets (CTS) by default, the setting we target first (ranked ladder, Global Challenges): a player
sees the foe's six species at team preview and the rest only as it is revealed. `Batch` itself always plays
closed sheets, even in the OTS format (it never calls `show_team_sheets`).

`open_sheets=True` (`--ots`, for supporting OTS later) fills the foe's items, abilities and moves from each
side's own view, which is always complete: recorded at team preview (all six mons) and refreshed at every later
decision of that side (Mega formes, items lost). Filled foe fields keep what the player has seen when it knows
it (an item knocked off stays "none"); moves are the sheet's four. `--ots` also checks the fill.
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import ninjask

FORMAT = "gen9championsvgc2026regmc"  # closed sheets; the OTS format's rules are the same
POOL = "data/teams_regmc.txt"
MON_IDS = 7  # species, item, ability, 4 moves (ninjask::encode::ids)

# Public state the 0.4 observation leaves out, rebuilt from each battle's log (PublicState).
PROTECT_WORDS = ("Protect", "Detect", "Shield", "Bunker", "Silk Trap", "Bulwark", "Obstruct")  # as stall_streaks
VOLATILES = ("substitute", "encore", "taunt", "perish", "leechseed", "disable", "yawn", "confusion", "healblock",
             "throatchop")
MON_X = 2 + len(VOLATILES)  # per mon: turns on the field, Protect streak, volatiles (perish = count / 3)
SIDE_CONDS = ("tailwind", "reflect", "lightscreen", "auroraveil")
FIELD = ("weather", "terrain", "trickroom", "gravity")
GLOB_X = 2 * len(SIDE_CONDS) + len(FIELD) + 1  # turns since each started (own side, foe side, field), turn number
TERRAINS = {"electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"}
SIDES, SLOTS = ("p1", "p2"), (("p1a", "p1b"), ("p2a", "p2b"))
VOL_INDEX = {v: k for k, v in enumerate(VOLATILES)}
ZEROS_X, ZEROS_V = [0.0] * MON_X, [0.0] * len(VOLATILES)
LOG_KINDS = ("|turn|", "|upkeep", "|move|", "|switch|", "|drag|", "|detailschange|", "|swap|", "|replace|", "|faint|",
             "|-singleturn|", "|-start|", "|-end|", "|-sidestart|", "|-sideend|", "|-fieldstart|", "|-fieldend|",
             "|-weather|")


def base_species(dex, fmt):
    """Array: species id (dex index + 1) -> its base species' id. Species Clause makes base species unique per
    team, so matching by it survives Mega Evolution and forme changes (Aegislash-Blade)."""
    # Keyed by the base's *name*: the base itself may not be in the dex (Floette-Eternal and Floette-Mega have
    # base Floette, which the format doesn't have).
    spec = ninjask.load(fmt).species
    names = dex.species_ids()
    code = {}
    base = np.zeros(len(names) + 1, np.int64)
    for k, name in enumerate(names, start=1):
        key = ninjask.to_id(spec.get(name, {}).get("baseSpecies") or name)
        base[k] = code.setdefault(key, len(code) + 1)
    return base


class PublicState:
    """What both players see in one battle but 0.4's observation leaves out; all species are *base* species.

    Each side's request order (what "switch N" means: the engine swaps the incoming mon's position with the
    active one's) comes from the choices themselves (`switched`, called by SelfPlay.step) plus drags and Ally
    Switch from the log: the log's switch lines can't be trusted for it, as Illusion shows a disguise. Own
    slots follow from the order (positions 0 and 1 are slots a and b). Foe slots come from the log's switch
    lines, i.e. what the player sees (a disguised Zoroark looks like its disguise). Also turns since each
    side/field condition started, and per slot turns on the field, Protect streak and volatiles.
    """

    def __init__(self, species_id, base):
        self.sid, self.base = species_id, base  # species id string -> dex index + 1; id -> base species id
        self.seen = 0  # log lines read
        self.turn = 0
        self.slot = {}  # 'p1a' -> base species seen entering there (popped when it faints)
        self.order = [[], []]  # per side: base species ids in request order (set at team preview)
        # per slot: turn it came in, Protect streak, volatiles (a list in VOLATILES order, already scaled)
        self.since, self.streak, self.vol = {}, {}, {}
        self.protected = set()  # slots that protected this turn
        self.reviving = [False, False]  # Revival Blessing used: the side's next "switch" choice revives instead
        self.cond = {}  # ('p1', 'tailwind') | 'weather' | 'terrain' | 'trickroom' | 'gravity' -> turn it started

    def species(self, details):
        return int(self.base[self.sid.get(ninjask.to_id(details.split(",")[0]), 0)])

    def switched(self, side, i, pos):
        """Side's slot i (0/1) switched to request position pos (0-based): the two trade positions."""
        if self.reviving[side]:
            self.reviving[side] = False
            return
        o = self.order[side]
        if i < len(o) and pos < len(o):
            o[i], o[pos] = o[pos], o[i]

    def _enter(self, slot, sp):
        self.slot[slot], self.since[slot], self.streak[slot], self.vol[slot] = sp, self.turn, 0, [0.0] * len(VOLATILES)

    def update(self, lines):
        for l in lines:
            if not l.startswith(LOG_KINDS):
                continue
            w = l.split("|")
            kind = w[1]
            if kind == "turn":
                self.turn = int(w[2])
                for s in self.slot:
                    self.streak[s] = self.streak.get(s, 0) + 1 if s in self.protected else 0
                self.protected.clear()
                self.reviving = [False, False]
            elif kind == "upkeep":  # a revive is chosen right after the move; a Revival Blessing that failed
                self.reviving = [False, False]  # mustn't swallow the forced switch after upkeep
            elif kind == "move":
                if len(w) > 3 and w[3] == "Revival Blessing":
                    self.reviving[int(w[2][1]) - 1] = True
            elif kind in ("-sidestart", "-sideend"):
                c = ninjask.to_id(w[3].removeprefix("move: "))
                if c in SIDE_CONDS:
                    if kind == "-sidestart":
                        self.cond[(w[2][:2], c)] = self.turn
                    else:
                        self.cond.pop((w[2][:2], c), None)
            elif kind in ("-fieldstart", "-fieldend"):
                c = ninjask.to_id(w[2].removeprefix("move: "))
                c = "terrain" if c in TERRAINS else c
                if c in FIELD:
                    if kind == "-fieldstart":
                        self.cond[c] = self.turn
                    else:
                        self.cond.pop(c, None)
            elif kind == "-weather":
                if w[2] == "none":
                    self.cond.pop("weather", None)
                elif "[upkeep]" not in l:
                    self.cond["weather"] = self.turn
            elif len(w) < 3 or ": " not in w[2]:
                continue
            elif kind == "switch":
                self._enter(w[2][:3], self.species(w[3]))
            elif kind == "drag":  # Roar, Dragon Tail, Red Card: not a choice, so the order changes here
                slot, sp = w[2][:3], self.species(w[3])
                o = self.order[int(slot[1]) - 1]
                if sp in o:
                    self.switched(int(slot[1]) - 1, "ab".index(slot[2]), o.index(sp))
                self._enter(slot, sp)
            elif kind == "replace":  # Illusion broken: the player now sees the real species
                self.slot[w[2][:3]] = self.species(w[3])
            elif kind == "swap":  # Ally Switch: the two slots trade mons and positions
                side = w[2][:2]
                a, b = side + "a", side + "b"
                for d in (self.slot, self.since, self.streak, self.vol):
                    va, vb = d.pop(a, None), d.pop(b, None)
                    if vb is not None:
                        d[a] = vb
                    if va is not None:
                        d[b] = va
                o = self.order[int(side[1]) - 1]
                if len(o) > 1:
                    o[0], o[1] = o[1], o[0]
            elif kind == "faint":
                for d in (self.slot, self.vol, self.streak):
                    d.pop(w[2][:3], None)
            elif kind == "-singleturn":
                if any(p in w[3] for p in PROTECT_WORDS):
                    self.protected.add(w[2][:3])
            elif kind in ("-start", "-end") and len(w) > 3:
                slot, v = w[2][:3], ninjask.to_id(w[3].removeprefix("move: "))
                vol = self.vol.setdefault(slot, [0.0] * len(VOLATILES))
                if v.startswith("perish"):
                    vol[VOL_INDEX["perish"]] = (int(v[6:] or 0) / 3) if kind == "-start" else 0.0
                elif v in VOL_INDEX:
                    vol[VOL_INDEX[v]] = 1.0 if kind == "-start" else 0.0

    def extras(self, side):
        """Plain lists for player `side` (0/1), matched against the view in bulk by `SelfPlay.observe`:
        base species in own a, own b, foe a, foe b (-2 = empty or fainted); those slots' MON_X features;
        base species at request positions 0-5 (-2 = none); GLOB_X counters."""
        o, turn = self.order[side], self.turn
        targets, feats = [-2, -2, -2, -2], []
        for v, sd in enumerate((side, 1 - side)):
            for i in range(2):
                slot = SLOTS[sd][i]
                if slot in self.slot:
                    targets[2 * v + i] = (o[i] if i < len(o) else -2) if v == 0 else self.slot[slot]
                    feats += [min(turn - self.since.get(slot, turn), 4) / 4, min(self.streak.get(slot, 0), 3) / 3]
                    feats += self.vol.get(slot) or ZEROS_V
                else:
                    feats += ZEROS_X
        order = (o + [-2] * 6)[:6]
        c = self.cond
        glob = [0 if (t := c.get((p, k))) is None else min(turn - t + 1, 8) / 8
                for p in (SIDES[side], SIDES[1 - side]) for k in SIDE_CONDS]
        glob += [0 if (t := c.get(k)) is None else min(turn - t + 1, 8) / 8 for k in FIELD]
        return targets, feats, order, glob + [min(turn, 30) / 30]


def load_pool(path=POOL, min_rating=0, meta_path=None):
    """Packed teams, one per line (fetch_regmc.py writes them best rating first)."""
    teams = [t for t in open(path, encoding="utf-8").read().splitlines() if t]
    if min_rating and meta_path:
        import json
        meta = json.load(open(meta_path, encoding="utf-8"))
        teams = [t for t, m in zip(teams, meta) if m["rating"] >= min_rating]
    return teams


@dataclass
class Decisions:
    """Every pending decision of an observe(), as numpy arrays (Batch.observe's layout)."""
    battle: np.ndarray   # (n,) battle slot
    side: np.ndarray     # (n,) 0 = p1, 1 = p2
    preview: np.ndarray  # (n,) bool: team preview
    obs: np.ndarray      # (n, OBS_LEN) float32
    ids: np.ndarray      # (n, 2, 6, MON_IDS) int16: own side, foe side (0 = unknown; sheet filled in if OTS)
    actions: np.ndarray  # (rows, ACTION_WIDTH) int8: legal choices of all decisions in turn
    offsets: np.ndarray  # (n + 1,) decision k's choices are actions[offsets[k]:offsets[k + 1]]
    # From the log (PublicState); view index = position in ids[:, v]:
    slots: np.ndarray    # (n, 2, 6) int8: 1 = slot a, 2 = slot b, 0 = not on the field
    pos2own: np.ndarray  # (n, 6) int8: own view index of request position j ("switch j+1"), -1 none
    mon_x: np.ndarray    # (n, 2, 6, MON_X) float32: turns on the field, Protect streak, volatiles
    glob_x: np.ndarray   # (n, GLOB_X) float32: turns since own/foe side conditions and field started, turn

    @property
    def n(self):
        return len(self.battle)

    def counts(self):
        return np.diff(self.offsets).astype(np.int64)


class SelfPlay:
    """n battles between random pool teams; finished battles restart with new teams.

    Loop: d = env.observe(); results = env.step(choices) with one index per decision, choices[k] in
    range(d.counts()[k]). step returns the battles that ended as (slot, winner side or None, turns).
    """

    def __init__(self, teams, n, seed=0, fmt=FORMAT, turn_limit=None, open_sheets=False):
        self.dex = ninjask.dex(fmt)
        self.teams, self.n, self.open_sheets = teams, n, open_sheets
        self.rng = np.random.default_rng(seed)
        self.batch = ninjask.Batch(self.dex, n)
        if turn_limit:
            self.batch.turn_limit = turn_limit
        self.species = self.dex.vocab()["species"] + 1  # ids are dex index + 1 (0 = unknown)
        self.species_id = {name: k + 1 for k, name in enumerate(self.dex.species_ids())}
        self.base = base_species(self.dex, fmt)
        # sheet[b, side, species id] = (item, ability, 4 moves) from that side's own view
        self.sheet = np.zeros((n, 2, self.species, MON_IDS - 1), np.int16)
        self.pairs = np.zeros((n, 2), np.int64)  # pool index of each side's team
        self.public = [None] * n
        self.last = None  # the last observe(), for step() to read team-preview choices
        self.games = 0
        self._reset(np.arange(n))

    def _reset(self, slots):
        slots = [int(i) for i in slots]
        picks = self.rng.integers(len(self.teams), size=(len(slots), 2))
        seeds = (self.games + np.arange(len(slots))).tolist()
        self.batch.reset_many(slots, [self.teams[a] for a, _ in picks], [self.teams[b] for _, b in picks], seeds)
        self.games += len(slots)
        self.pairs[slots] = picks
        self.sheet[slots] = 0
        for i in slots:
            self.public[i] = PublicState(self.species_id, self.base)

    def observe(self):
        o = self.batch.observe()
        n = o["n"]
        who = np.frombuffer(o["who"], np.uint32).reshape(n, 2).astype(np.int64)
        ids = np.frombuffer(o["ids"], np.int16).reshape(n, 2, 6, MON_IDS)
        b, s = who[:, 0], who[:, 1]
        if self.open_sheets:
            ids = self._fill_sheets(ids.copy(), b, s)
        for i in np.unique(b):  # read each deciding battle's new log lines (a copy of the battle: ~10-26 us)
            st = self.public[i]
            battle = self.batch.battle(int(i))
            st.update(battle.log("omniscient", st.seen))
            st.seen = battle.log_len()
        ex = [self.public[i].extras(j) for i, j in zip(b.tolist(), s.tolist())]
        targets = np.array([e[0] for e in ex], np.int64).reshape(n, 2, 2)
        feats = np.array([e[1] for e in ex], np.float32).reshape(n, 2, 2, MON_X)
        order = np.array([e[2] for e in ex], np.int64)
        glob_x = np.array([e[3] for e in ex], np.float32).reshape(n, GLOB_X)
        sp = ids[:, :, :, 0]
        base = np.where(sp > 0, self.base[sp], -1)  # (n, 2, 6) base species of each view mon; -1 unknown
        hit = base[:, :, :, None] == targets[:, :, None, :]  # (n, 2, 6 mons, 2 slots)
        slots = (hit * np.array([1, 2])).max(axis=3).astype(np.int8)
        mon_x = np.einsum("nvms,nvsx->nvmx", hit.astype(np.float32), feats)
        at = base[:, 0, :, None] == order[:, None, :]  # (n, 6 mons, 6 positions)
        pos2own = np.where(at.any(axis=1), at.argmax(axis=1), -1).astype(np.int8)
        self.last = Decisions(
            battle=b, side=s, preview=np.frombuffer(o["preview"], np.uint8).astype(bool),
            obs=np.frombuffer(o["obs"], np.float32).reshape(n, ninjask.OBS_LEN), ids=ids,
            actions=np.frombuffer(o["actions"], np.int8).reshape(-1, ninjask.ACTION_WIDTH),
            offsets=np.frombuffer(o["offsets"], np.uint32).astype(np.int64),
            slots=slots, pos2own=pos2own, mon_x=mon_x, glob_x=glob_x)
        return self.last

    def _fill_sheets(self, ids, b, s):
        """OTS: the foe's unknown fields from the foe's own (complete) view, matched by species."""
        # 1) every decider's own view is complete: refresh its sheet (all six mons at preview)
        own = ids[:, 0]
        sp = own[:, :, 0]
        rows = np.broadcast_to(b[:, None], sp.shape), np.broadcast_to(s[:, None], sp.shape)
        known = sp > 0
        self.sheet[rows[0][known], rows[1][known], sp[known]] = own[:, :, 1:][known]
        # 2) fill the foe's unknown fields from the foe's sheet, by species (unique per team)
        foe = ids[:, 1]
        fsp = foe[:, :, 0]
        sheet = self.sheet[b[:, None], 1 - s[:, None], fsp]  # (n, 6, 6)
        have = (sheet != 0).any(axis=2) & (fsp > 0)
        item, ability = foe[:, :, 1], foe[:, :, 2]
        foe[:, :, 1] = np.where(have & (item == 0), sheet[:, :, 0], item)
        foe[:, :, 2] = np.where(have & (ability == 0), sheet[:, :, 1], ability)
        foe[:, :, 3:] = np.where(have[:, :, None], sheet[:, :, 2:], foe[:, :, 3:])
        return ids

    def step(self, choices):
        """Apply one choice index per decision of the last observe(); restart finished battles.

        A refused choice (a hidden disable or trap, as in Showdown) stays pending and shows up again.
        Returns [(slot, winner side 0/1 or None for a draw, turns)] for the battles that ended.
        """
        d = self.last
        res = np.frombuffer(self.batch.step([int(c) for c in choices]), np.uint8)
        self.refused = (res & 1) == 1  # per decision of the last observe: not applied, asked again
        for k in np.flatnonzero(~self.refused):  # accepted choices: they move mons between request positions
            st, side = self.public[d.battle[k]], int(d.side[k])
            row = d.actions[d.offsets[k] + int(choices[k])]
            if d.preview[k]:  # the order starts as the preview choice: leads first
                team = st.base[d.ids[k, 0, :, 0]]  # own view at preview: all six, in team order
                st.order[side] = [int(team[p - 1]) for p in row if p > 0]
                continue
            for i in range(2):
                if row[4 * i] == 2:  # kind switch, index = 1-based request position
                    st.switched(side, i, int(row[4 * i + 1]) - 1)
        done = np.flatnonzero(np.frombuffer(self.batch.dones(), np.uint8))
        ended = []
        for i in done:
            w = self.batch.winner(int(i))
            ended.append((int(i), {"p1": 0, "p2": 1}.get(w), self.batch.turn(int(i))))
        if len(done):
            self._reset(done)
        return ended


def check_sheets(d):
    """Where both players of a battle decide at once, the filled foe block must agree with the foe's own view
    on every mon the foe has brought (species match): returns (mons checked, mismatches)."""
    idx = {(int(b), int(s)): k for k, (b, s) in enumerate(zip(d.battle, d.side))}
    checked = bad = 0
    for (b, s), k in idx.items():
        j = idx.get((b, 1 - s))
        if j is None:
            continue
        mine = {int(m[0]): m for m in d.ids[k, 1] if m[0] > 0}
        for m in d.ids[j, 0]:
            if m[0] > 0 and int(m[0]) in mine:
                f = mine[int(m[0])]
                checked += 1
                bad += not (f[1] == m[1] and f[2] == m[2] and sorted(f[3:]) == sorted(m[3:]))
    return checked, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--secs", type=float, default=20)
    ap.add_argument("--ots", action="store_true", help="open team sheets (fill the foe's sheet) and check the fill")
    args = ap.parse_args()
    teams = load_pool()
    env = SelfPlay(teams, args.n, open_sheets=args.ots)
    rng = np.random.default_rng(1)
    games, decisions, wins, draws, turns = 0, 0, np.zeros(2), 0, []
    checked = bad = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.secs:
        d = env.observe()
        if args.ots and games < 2000:  # the sheet check is pure Python: only on the first stretch
            c, x = check_sheets(d)
            checked, bad = checked + c, bad + x
        decisions += d.n
        for _, w, t in env.step((rng.random(d.n) * d.counts()).astype(np.int64)):
            games += 1
            turns.append(t)
            if w is None:
                draws += 1
            else:
                wins[w] += 1
    el = time.perf_counter() - t0
    print(f"{len(teams)} pool teams, {args.n} battles: {games / el:.0f} games/s, {decisions / el:.0f} decisions/s; "
          f"p1 wins {wins[0] / games:.1%}, draws {draws / games:.1%}, turns median {int(np.median(turns))}")
    if args.ots:
        print(f"open sheets: {checked} foe mons checked against the foe's own view, {bad} mismatches")


if __name__ == "__main__":
    main()
