"""Run: .venv/bin/python test_analyze.py"""

import json

import ninjask
import analyze as a

L = json.load(open(a.LOGS))
log = L["gen9championsvgc2026regmb-2635876837"][1]

# Blaziken protected on turn 2, so its turn-3 Protect should fail ~2/3 of the time.
assert a.stall_streaks(log.split("\n"), 3) == {"p1a: Blaziken": 1}
b = a.load_position(log, 3, ninjask.dex(a.FORMAT))
assert json.loads(b.projection())["sides"][0]["pokemon"][0]["volatiles"]["stall"]["counter"] == 3
print("ok")

# Both sides set Tailwind on turn 1 of this game: 3 turns left each at the start of turn 2.
tw = a.load_position(L["gen9championsvgc2026regmb-2635668924"][1], 2, ninjask.dex(a.FORMAT))
assert {s: c["tailwind"] for s, c in a.side_conditions(tw).items()} == {"p1": 3, "p2": 3}
print("ok tailwind")

# Boosts rebuilt from the turn's log must match the engine's (Blaziken's Speed Boost, Metagross's +1/+1, etc).
from ninjask.choices import joint_choices
root = a.active_boosts(b)
A = joint_choices(json.loads(b.request("p1")), game_type="doubles")
B = joint_choices(json.loads(b.request("p2")), game_type="doubles")
for i, (x, y) in enumerate((x, y) for x in A[::6] for y in B[::6]):
    c = b.copy(); c.set_logging(True); c.reseed(i); k = c.log_len()
    try:
        c.choose("p1", x); c.choose("p2", y)
    except ValueError:
        continue
    a.finish_turn(c, k)
    if not c.ended:
        assert a.apply_boost_log(root, c.log("omniscient", k)) == a.active_boosts(c), (x, y)
print("ok boosts")

# Side conditions per cell (root counted down plus what the log shows set or ended) must match the engine:
# Tailwind and Light Screen set fresh on turn 1, both Tailwinds expiring on turn 4 (Light Screen still up),
# Aurora Veil counting down, Psychic Fangs (screen remover).
def check_conds(pos, step):
    root = a.side_conditions(pos)
    aged = {s: {c: max(d - 1, 0) for c, d in cs.items()} for s, cs in root.items()}
    A = joint_choices(json.loads(pos.request("p1")), game_type="doubles")
    B = joint_choices(json.loads(pos.request("p2")), game_type="doubles")
    n = changed = 0
    for i, (x, y) in enumerate((x, y) for x in A[::step] for y in B[::step]):
        c = pos.copy(); c.set_logging(True); c.reseed(i); k = c.log_len()
        try:
            c.choose("p1", x); c.choose("p2", y)
        except ValueError:
            continue
        a.finish_turn(c, k)
        if not c.ended:
            got = a.turn_conds(c, c.log("omniscient", k), aged)
            assert got == a.side_conditions(c), (x, y, got)
            n, changed = n + 1, changed + (got != aged)
    return n, changed

t1 = a.load_position(L["gen9championsvgc2026regmb-2635668924"][1], 1, ninjask.dex(a.FORMAT))
t4 = a.load_position(L["gen9championsvgc2026regmb-2635668924"][1], 4, ninjask.dex(a.FORMAT))
v = a.load_position(L["gen9championsvgc2026regmb-2635726495"][1], 3, ninjask.dex(a.FORMAT))
counts = [check_conds(pos, step) for pos, step in ((t1, 2), (t4, 1), (v, 2), (b, 3))]
assert all(n > 100 for n, _ in counts) and counts[0][1] > 50, counts
print(f"ok conds {counts}")

# HP and status rebuilt from the turn's log must match the engine's requests, including Regenerator's heal
# (TheFatrat, 130/202, switches out in every cell of the second position).
def req_hp(battle):
    return {(s, p["ident"].split(": ", 1)[1]): a.hp_field(p["condition"])
            for s in ("p1", "p2") for p in json.loads(battle.request(s))["side"]["pokemon"]}

regen = a.load_position(L["gen9championsvgc2026regmb-2635528532"][1], 2, ninjask.dex(a.FORMAT))
n = 0
for pos, step in ((b, 4), (regen, 1)):
    mons = a.snapshot(pos)[2]
    A = joint_choices(json.loads(pos.request("p1")), game_type="doubles")
    B = joint_choices(json.loads(pos.request("p2")), game_type="doubles")
    if pos is regen:
        A = [x for x in A if x.endswith("switch 3")]
    for i, (x, y) in enumerate((x, y) for x in A[::step] for y in B[::step]):
        c = pos.copy(); c.set_logging(True); c.reseed(i); k = c.log_len()
        try:
            c.choose("p1", x); c.choose("p2", y)
        except ValueError:
            continue
        a.finish_turn(c, k)
        if not c.ended and c.request("p1") and c.request("p2"):
            assert a.apply_hp_log(mons, c.log("omniscient", k)) == req_hp(c), (x, y)
            n += 1
assert n > 300, n
print(f"ok hp ({n} cells)")

# Mega Gengar's Shadow Tag makes p1's switches illegal; that must drop only those rows, not every column.
g = a.load_position(L["gen9championsvgc2026regmb-2634992509"][1], 2, ninjask.dex(a.FORMAT))
A, B, F = a.payoff_matrix(g, "p1", "p2", 1)
assert len(A) == 35 and len(B) == 70 and not any("switch" in x for x in A), (len(A), len(B))
print("ok trapped")

# Parting Shot pauses the turn for a switch-in; finish_turn must run it to the end (upkeep, screens tick down).
root = a.side_conditions(v)
assert root["p1"]["auroraveil"] == 3, root
A = joint_choices(json.loads(v.request("p1")), game_type="doubles")
B = joint_choices(json.loads(v.request("p2")), game_type="doubles")
x = next(x for x in A if a.name_choice(v, "p1", x).startswith("Blizzard | Fake Out"))
y = next(y for y in B if "Parting Shot" in a.name_choice(v, "p2", y))
c = v.copy(); c.set_logging(True); c.reseed(0); k = c.log_len()
c.choose("p1", x); c.choose("p2", y)
assert c.needs_choice("p2")  # paused mid-turn
a.finish_turn(c, k)
assert c.turn == v.turn + 1 and a.side_conditions(c)["p1"]["auroraveil"] == 2
print("ok pivot + screens")
