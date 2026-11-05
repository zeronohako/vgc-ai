# vgc-ai

## Goal
A dashboard that analyzes a Pokémon VGC position (4 active Pokémon) and shows the best moves, plus which 4 to bring and 2 to lead. **Not** a ladder bot. Background research: `vgc-bot-research-summary.md`.

Build order: search engine → better evaluator → bring/lead selection → UI. Python first; move hot loops to Rust only when timings demand it.

## Setup
- `.venv` (Python 3.14) with `ninjask==0.3.0` and `numpy` (`requirements.txt`). Run with `.venv/bin/python` (Linux/macOS) or `.venv\Scripts\python` (Windows); commands below use the Linux form. Windows setup: `py -3.14 -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt`. Run from the repo root (`LOGS` is a relative path).
- `data/logs_regmb.json`: 324 OTS replays (Reg M-B) from the VGC-Bench dataset (HF `cameronangliss/vgc-battle-logs`). Dict `replay_id -> [timestamp, log_text]`. The dataset only covers Reg M-A/M-B, so the format is `gen9championsvgc2026regmb` for now.

## Status (2026-11-05)
`analyze.py` works end to end:
```
.venv/bin/python analyze.py <replay_id> <turn> [--side p1|p2] [--seeds 16] [--top 5]
.venv/bin/python test_analyze.py                  # loader/feature checks, ~1 min
.venv/bin/python check_positions.py [--grid] [--cache FILE.pkl]   # agreement with humans, ~6 min
```
Pipeline: replay log → position at start of `<turn>` in Ninjask → every joint action for both sides → each pair played with N seeds → feature vector per cell → score = `F @ WEIGHTS` → regret matching → ranked moves + opponent's likely plays.

**Evaluator** (`FEATURES`/`WEIGHTS` in `analyze.py`, my side minus foe): hp (per mon 0.5 alive + 0.5·HP fraction; ±10 for a win) 1.0, Tailwind turns left 0.15, boost stages 0.08 (all stats equal), screens (Reflect + Light Screen turns, Veil = both) 0, protect (Protect-type moves chosen this turn) 0, status per mon slp/frz −0.3, par/brn/tox −0.15, psn −0.1. Ignores Trick Room, Encore/Perish/redirection value, and anything past one turn.

**Agreement with humans** (2026-11-05, `check_positions.py --grid`: 40 random positions, seed 0, plus 3 dev; 42 simulated, 84 decisions). "Worse" = share of our options ranked strictly above the human's (0 = our top pick, 0.5 = random). Fresh: 0.245 (human = our top pick 10/78, top 10% 31/78); dev 0.13. A coordinate search over the weights (tuned on dev + even fresh, tested on odd fresh) kept the current weights everywhere: train 0.240, held-out test 0.225. A **Protect penalty makes it worse** (−0.1: 0.244, −0.3: 0.276, −0.5: 0.295): humans Protect often too, so the one-turn horizon is the issue, not a missing flat cost. Screens weight is flat (0–0.1 all ≈ 0.241). Tailwind 0 is clearly worse (0.262); boost and status are flat-ish around the current values. **Weight tuning is done**; gains now need new information (horizon, Trick Room, better spreads). Players are mostly unrated Bo3 ladder players, not experts.
- One position skipped: the human switched to `Sinistcha-Masterpiece` but `name_choice` prints `switch Sinistcha`; the switch match in `check_positions.matches` should compare by base species.

### What we learned
- **Position loading:** `Tracker.world()` works on spectator logs, but the tracker's *own* side is garbage without requests (HP as %, wrong stats). Workaround in `load_position`: build each side from the *other* player's view (where it's the foe), merge the two projections, `Battle.from_projection`. The 4 brought Pokémon come from the whole log via the `{"brought": [idx]}` assumption. Spreads are hidden in OTS; Ninjask's default guess is used.
- **Lost state on load:** the projection drops the `stall` volatile, so `stall_streaks` rebuilds it from the log (`{"counter": 3**n, "duration": 1}`); without it a repeat Protect always works.
- **Speed:** `copy + reseed + choose×2` ≈ 22 µs; `request` + `json.loads` ≈ 140 µs per side dominates. `projection()` + `json.loads` ≈ 900 µs, so avoid it per cell: side conditions (Tailwind, screens) are counted down from the root (re-reading only when a fresh one is set or a screen remover is used; 0/2870 mismatches), and boosts are rebuilt from the turn's log (`apply_boost_log`, logging on costs +17 µs). Both checked against the engine with 0 mismatches. Full matrix 5–45 s per position; a top-8 cut per side would give ~50× once a policy can rank actions.
- **Pivot moves:** U-turn/Volt Switch/Parting Shot pause the turn for a switch-in; `finish_turn` answers with the first legal switch-in so cells are scored after upkeep (before this, every pivot cell was scored mid-turn).
- **Illegal actions:** `joint_choices` lists switches even when trapped (Shadow Tag etc.). `payoff_matrix` drops rows/columns rejected everywhere before mixed cells; dropping on any rejection used to wipe the whole matrix.

## Next steps
1. **Richer heuristic**: Tailwind, boosts, status, screens, protect done; weights tuned on 42 positions (no change). Next: a longer horizon (2-ply on a top-k cut, or short rollouts), the real fix for Protect/Encore/Perish; Trick Room; fix the Sinistcha-Masterpiece switch match. Cache for re-tuning: `--cache FILE.pkl` (~25 min to simulate 43 positions).
2. **Value network** (months, Phases 1–4 of the research doc): self-play on Ninjask; the heuristic is the baseline it must beat.
3. **Bring/lead:** 15 bring sets × 6 leads = 90 options per side; fill a 90×90 matrix with simulated games (needs a decent playout policy).
4. **Dashboard UI** on top of `analyze.py`.
5. Reg M-C positions need sources other than VGC-Bench (own games or other replays).
