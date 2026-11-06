"""Run: .venv/bin/python test_selfplay.py   (~20 s)

The log-derived PublicState must agree with the engine during random self-play on the pool: each side's request
order, which mon is in slot a/b (own side and foe side), what "switch N" points to, and every turn counter being
present exactly when the observation's flag is.
"""

import json
import random

import numpy as np
import ninjask

import selfplay as sp

env = sp.SelfPlay(sp.load_pool(), 96, seed=7)
names = {int(env.base[k + 1]): n for k, n in enumerate(env.dex.species_ids())}  # base code -> a species name
rng = random.Random(0)
# observation flags (ninjask src/encode.rs): 6 mons x 19 floats per side, then 6 side conditions
OWN_CONDS, FOE_CONDS, FIELD0 = 114, 120 + 114, 240
move_id = {m: k + 1 for k, m in enumerate(env.dex.move_ids())}  # ids are dex index + 1
checked = {"order": 0, "own slots": 0, "move order": 0, "foe slots": 0, "switch targets": 0, "counters": 0}
bad = {k: [] for k in checked}


def base(details):
    """Base species id of a request entry: the tracker matches base species (formes, Mega Evolution)."""
    return int(env.base[env.species_id[ninjask.to_id(details.split(",")[0])]])


def actives(req):
    """{0: base species id, 1: ...} of the request's living active mons (request order = slot order)."""
    return {i: base(p["details"]) for i, p in enumerate(req["side"]["pokemon"][:2])
            if p.get("active") and not p["condition"].endswith("fnt")}


megas = illusions = 0
for step in range(600):
    d = env.observe()
    for k in range(d.n):
        if d.preview[k] or rng.random() > 0.15:
            continue
        b, s = int(d.battle[k]), int(d.side[k])
        battle = env.batch.battle(b)
        req = json.loads(battle.request(("p1", "p2")[s]))
        truth = [base(p["details"]) for p in req["side"]["pokemon"]]
        megas += sum("mega" in ninjask.to_id(p["details"].split(",")[0]) for p in req["side"]["pokemon"][:2])
        checked["order"] += 1
        if env.public[b].order[s] != truth:
            bad["order"].append((step, b, s, [names[x] for x in env.public[b].order[s]], [names[x] for x in truth],
                                 battle.log("omniscient", max(0, battle.log_len() - 25))))
        own = env.base[d.ids[k, 0, :, 0]]
        for i, spc in actives(req).items():
            checked["own slots"] += 1
            hit = np.flatnonzero(d.slots[k, 0] == i + 1)
            if len(hit) != 1 or own[hit[0]] != spc:
                bad["own slots"].append((step, b, s, i, names[spc], [names.get(int(x)) for x in own], d.slots[k, 0].tolist()))
                continue
            # "move j" of slot i means the j-th move of the request; the view must list the moves in that order
            moves = req["active"][i]["moves"] if i < len(req.get("active", [])) else []
            if len(moves) == 4:  # locked moves (recharge, Struggle) come as one entry
                checked["move order"] += 1
                want = [move_id.get(m["id"], -1) for m in moves]
                if d.ids[k, 0, hit[0], 3:].tolist() != want:
                    bad["move order"].append((step, b, s, i, want, d.ids[k, 0, hit[0], 3:].tolist()))
        foe_req = battle.request(("p2", "p1")[s])
        if foe_req:
            foe = env.base[d.ids[k, 1, :, 0]]
            for i, spc in actives(json.loads(foe_req)).items():
                checked["foe slots"] += 1
                hit = np.flatnonzero(d.slots[k, 1] == i + 1)
                if len(hit) != 1 or foe[hit[0]] != spc:
                    # Illusion: the player sees the disguise, as it should under CTS (a Zoroark disguised as
                    # its own Malamar even shows two Malamars on the field)
                    if any("zoroark" in names.get(int(x), "") for x in foe):
                        illusions += 1
                        continue
                    bad["foe slots"].append((step, b, s, i, names[spc], [names.get(int(x)) for x in foe]))
        for j, spc in enumerate(truth):
            checked["switch targets"] += 1
            m = d.pos2own[k, j]
            if m < 0 or own[m] != spc:
                bad["switch targets"].append((step, b, s, j))
        flags = np.r_[d.obs[k, OWN_CONDS:OWN_CONDS + 4], d.obs[k, FOE_CONDS:FOE_CONDS + 4],
                      d.obs[k, FIELD0:FIELD0 + 4].max(), d.obs[k, FIELD0 + 4:FIELD0 + 8].max(), d.obs[k, FIELD0 + 8:FIELD0 + 10]]
        checked["counters"] += 1
        if not np.array_equal(flags > 0, d.glob_x[k, :-1] > 0):
            bad["counters"].append((step, b, s, (flags > 0).astype(int).tolist(), (d.glob_x[k, :-1] > 0).astype(int).tolist()))
    env.step((np.random.default_rng(step).random(d.n) * d.counts()).astype(np.int64))

for name in checked:
    print(f"{name:<15} {checked[name]:6d} checked, {len(bad[name])} wrong" + (f"; first: {bad[name][0]}" if bad[name] else ""))
print(f"(active Mega formes in the checked requests: {megas}; foe slots fooled by Illusion: {illusions})")
assert all(checked.values()) and not any(bad.values())
print("ok")
