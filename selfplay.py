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
        # sheet[b, side, species id] = (item, ability, 4 moves) from that side's own view
        self.sheet = np.zeros((n, 2, self.species, MON_IDS - 1), np.int16)
        self.pairs = np.zeros((n, 2), np.int64)  # pool index of each side's team
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

    def observe(self):
        o = self.batch.observe()
        n = o["n"]
        who = np.frombuffer(o["who"], np.uint32).reshape(n, 2).astype(np.int64)
        ids = np.frombuffer(o["ids"], np.int16).reshape(n, 2, 6, MON_IDS)
        b, s = who[:, 0], who[:, 1]
        if self.open_sheets:
            ids = self._fill_sheets(ids.copy(), b, s)
        return Decisions(
            battle=b, side=s, preview=np.frombuffer(o["preview"], np.uint8).astype(bool),
            obs=np.frombuffer(o["obs"], np.float32).reshape(n, ninjask.OBS_LEN), ids=ids,
            actions=np.frombuffer(o["actions"], np.int8).reshape(-1, ninjask.ACTION_WIDTH),
            offsets=np.frombuffer(o["offsets"], np.uint32).astype(np.int64))

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
        self.batch.step([int(c) for c in choices])
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
