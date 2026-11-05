# Pokémon VGC Bot: Research Summary and Build Plan

Summary of a research conversation (as of November 4, 2026) about how competitive Pokémon AI works, how the VGC bot **mikumiku37** reached #1 on Showdown, and how to build a similar bot on a personal machine using the open-source **Ninjask** simulator. Written as context for coding in Claude Code.

---

## 1. Why Pokémon is hard for AI

Chess and Go have full information and alternating turns. Pokémon has neither. The opponent's moves, items, abilities and stat spreads are hidden until revealed; damage rolls, crits, accuracy and secondary effects add chance to every turn; and both players choose at the same time, so being predictable gets punished.

VGC (doubles) is harder still. Each player picks a joint action for two active Pokémon (move plus target per slot), so menus run into the hundreds, and each joint action can resolve into hundreds of random outcomes. VGC-Bench estimates roughly 10^139 possible team configurations, far beyond chess, Go, poker, StarCraft or Dota. Open Team Sheets (OTS) remove much of the hidden information, which makes OTS the easier setting for bots.

## 2. The four parts of a strong bot

Every top bot combines the same building blocks.

| Part | What it does | Plain-language version |
|---|---|---|
| Simulator | Runs the exact battle rules privately, fast | A practice room inside the bot's head |
| Neural network | Policy head (which actions look good) and value head (who is winning) | Gut feeling |
| Search | Tries candidate actions in the simulator before choosing | Thinking ahead |
| Hidden-info inference | Narrows down the opponent's sets from clues | Detective work |

Two ways to train the network: **imitation** (behavior cloning from human replays) and **self-play reinforcement learning** (play millions of games against itself, usually with PPO). Many projects do imitation first, then self-play. The strongest 2026 bots use pure self-play.

Search matters enormously. PokaiTrainer's paper reports its network alone loses even to a shallow heuristic search. mikumiku37's network alone peaked at 1747 Elo; adding search took it to #1.

## 3. Landscape of notable bots

| Bot | Format | Method | Result | Open source? |
|---|---|---|---|---|
| **mikumiku37** (Shao) | VGC Reg M-C (Champions) | Self-play PPO transformer + one-turn matrix search | #1 on Reg M-C ladder, Oct 1 2026, 1857 Elo, 81.2% GXE | No |
| Nessie123 | VGC Reg M-C Bo3 (OTS) | Solves root simultaneous game, selectively deepens leaves | Topped Reg M-C Bo3 ladder, Oct 2026 | Unknown |
| PokaiTrainer (Max Yu) | VGC, OTS only | Poker-style equilibrium search (CFR over public belief states, Student of Games / ReBeL lineage), custom Rust engine | 59% of 150 Bo3 sets vs ~1320 field; 1350–1400 band; peak 1492 (top 500) | Paper only (arXiv 2608.29197) |
| VGC-Bench (UT Austin) | VGC, OTS | Benchmark: behavior cloning, PPO self-play, fictitious play, double oracle, LLM and heuristic baselines; 700k+ OTS battle logs | Beat a pro only in a single-team mirror; generalizing across many teams is unsolved | Yes |
| Jaxcalibur | Gen 9 Random Battles (singles) | Self-play RL + pUCT tree search (AlphaGo Zero style), JAX + poke-env | #1, peak 2557 Elo, 95.3 GXE (Aug 2026) | No (public technical write-up) |
| MissingNo.Zero | Random Battles gens 1, 2, 4, 6, 7 | Pure self-play PPO, ~3M-param net, short lookahead at play time, no human data | #1 on five random-battle ladders, Sept–Oct 2026; ~$2,006 total GPU rental | No |
| Foul Play (pmariglia) | Singles, many gens | Root-parallel MCTS on poke-engine (Rust), strong set prediction | Peak ~2341 Elo; won Gen 9 OU track of PokéAgent Challenge (NeurIPS 2025) | Yes (GPL-3.0) |
| Metamon (UT Austin) | Singles, gens 1–4 | Offline RL on 3.5M reconstructed human battles, transformers up to 200M params, no search | Top 10% of active players | Yes |
| PokéChamp | Gen 9 OU | LLM inside depth-limited minimax (LLM proposes moves, models opponent, scores states) | Projected 1300–1500 Elo (top 30–10%) | Yes |

**Board-analysis tools** are a different category: advisors, not players. Pokémon Battle Scope (HEROZ + The Pokémon Company, built on shogi-AI tech) shows likely strong next moves on Japan Championships broadcasts (2024–2026). Most fan tools are damage calcs plus usage stats; some just generate prompts for chatbots.

Glossary: **Elo** is the ladder rating; **GXE** is Showdown's estimate of the chance to beat a random ladder player; **OTS** is Open Team Sheets.

## 4. mikumiku37 in detail (the target methodology)

Source: Shao's Smogon thread "mikumiku37: a self-play VGC bot reached #1 on the VGC Reg M-C ladder" (Oct 2, 2026) plus his Oct 7 Q&A reply. A full technical write-up was promised but not yet published as of Oct 28. He says part of the design was adapted from the public Jaxcalibur write-up.

### 4.1 Model and inputs

| Item | Detail |
|---|---|
| Architecture | Small transformer, ~8.7M parameters, trained from scratch |
| Observation | Battle from its own side + static dex data (types, base stats, move data) only |
| Deliberately excluded | Damage calculator, Smogon usage stats, speed resolver |
| Joint action output | Each slot's action includes its target. Slot 1 predicted first; slot 2 predicted conditioned on slot 1 via a pairwise term; legality masks on the pair |
| Reward | Terminal only (win/loss) |
| Human data | None |

### 4.2 Training

| Item | Detail |
|---|---|
| Algorithm | PPO self-play |
| Opponents | ~50% current policy, ~50% past snapshots weighted toward ones it still struggles against (PFSP-style). Never ablated against pure self-play |
| Team pool | ~1,260 public tournament teams from Reg M-B and M-C (some spreads guessed) |
| Simulator | Custom Rust engine, ~650x faster than Showdown, ~7k random battles/s on one core |
| Batch | 8,192 battles in parallel, 32 steps each per update (~262k samples per update) |
| Collection throughput | ~100k steps/s including network inference |
| Hardware | One RTX 5090; PPO gradient update ~70% of wall time (compute-bound on learning, not on the sim) |
| Training time | 48.3 hours; the #1 checkpoint had played ~330M training games; network not saturated |
| Lesson | His earlier Showdown-based attempt got ~700 battles/minute with text-parsed observations and kept ~0.5% of the GPU busy. A fast simulator is the key enabler |

Search was **not** used during training; only at play time.

### 4.3 Play-time search (one-turn matrix game)

For each decision:

1. Sample **16 worlds**: plausible versions of the opponent's hidden information.
2. In each world, take the policy's **top 8 joint actions** for us and the **top 8 replies** for the opponent, forming an 8×8 payoff table.
3. Fill each cell: apply both joint actions in the engine, then evaluate the next position with the value network (no rollout). Chance is handled by stepping the real engine with **several RNG seeds per cell** and averaging the value over seeds and worlds.
4. Solve the table with **regret matching**, blending worst-case with expected value, producing a **mixed strategy** (probabilities over actions).

Seeds per cell: 4 caused nonsensical plays from outlier worlds; the public version used 16; Shao says 64–128 is ideal. Endgame deepening was tried: ~51.4% win rate vs baseline, not worth the extra compute.

Contrast: Jaxcalibur runs pUCT tree search four plies deep (~two turns) with 20,480 rollouts across 32 sampled worlds. Shao chose a wide one-turn matrix because doubles produces far more joint actions than singles.

### 4.4 Results and observed behavior

Network alone peaked at 1747 over 100 games. With search: 248–110 (69.3%), reached #1 in its 358th game, rotating among the ten teams that performed best in training. Peak: 1857 Elo, 81.2% GXE, 1775 ± 25 Glicko.

Player feedback: a Baltimore Regionals top-16 player went 3–9 and said it punished team weaknesses with fundamentally sound play rather than tricks. A Perish Trap bug in search was found and fixed. One player claimed it rarely anticipates straightforward attacks (e.g., Kingambit), though he was 1–3 against it.

## 5. Ninjask: the open-source simulator (checked and tested)

Repo: https://github.com/dcabatin/Ninjask · `pip install ninjask` · MIT license · Rust engine with Python bindings (PyO3/maturin).

### 5.1 Claims from the README

The simulator is a port of Showdown's battle code at a pinned commit. In reference mode it uses Showdown's RNG and makes identical draws, reportedly matching Showdown on 1,000/1,000 seeded games with random legal teams and 1,000/1,000 with real tournament teams, with CI replaying 200 games on every change. Normal mode uses a faster RNG (xoshiro256++) whose outcome distributions are checked statistically against Showdown.

Speed: ~7,000 random games/s (70k–85k turns/s) on one core with logging off, vs Showdown's ~200 turns/s on the same machine (~250x). Coverage for Reg M-C: all 511 moves, all 166 items, 221 of 224 abilities (the other three belong to Pokémon not legal in the format). `Dex.team_problems(team)` lists anything unsupported; nothing is silently approximated.

Agent tooling included:

| Tool | What it gives you |
|---|---|
| `Tracker` | One player's view built from protocol lines; also reads Showdown's live logs, so training and live play produce the same observations |
| Worlds (`Tracker.world(assumptions)`) | A `Battle` consistent with one player's view, with the foe's hidden sets filled from your assumptions. This is the ingredient for mikumiku37-style sampled worlds |
| Exact calc (`Battle.damage`, `.order`, `.speeds`) | All 16 rolls, crit rolls, KO chances, turn order with priority, Trick Room, Tailwind. Author's test: 0 of 99,176 damages fell outside the predicted distribution |
| `Battle.copy()` | Cheap independent copy for search |
| OTS support | Format `gen9championsvgc2026regmcots` |

Not included: an agent, a search algorithm, or a UI.

### 5.2 What I verified by running it (sandbox, 1 CPU core, Python 3.12)

Installed `ninjask==0.3.0` from PyPI without issues. Bundled formats: `gen9championsvgc2026regmc`, `gen9championsvgc2026regmcots`, `gen9championsvgc2026regmb`, `gen9championsdoublescustomgame`. `ninjask.verification_teams()` returns 4 test teams.

| Measurement | Result |
|---|---|
| Full games via `play_engine_game` with Python `RandomPolicy` | ~195 games/s, ~2,179 turns/s |
| `Battle.request(side)` | ~51 µs per call |
| `json.loads` on a request | ~15 µs |
| `Battle.copy()` | ~2.4 µs |
| `Battle.damage("p1a", move, "p2a")` | Returns JSON with 16 normal and 16 crit rolls, crit chance, accuracy, hit counts |

**Key finding:** the 7k games/s headline is the Rust-level speed. From a Python loop, every decision goes through a JSON request plus Python choice enumeration, giving ~195 games/s per core. That is still ~10x faster than Showdown, fine for prototyping, but ~35x below the Rust speed. For scale: 330M games would take ~19.6 days on one core from Python (before any network inference) vs ~13 hours in Rust.

The Python `Battle` exposes: `choose, copy, damage, ended, from_projection, log, log_len, needs_choice, order, projection, reference, reseed, rng_seed, rng_trace, set_logging, set_rng_trace, show_team_sheets, speeds, turn, winner, winner_side`. The fast Rust helpers `joint_choices`, `sample_joint_choice` and `random_choice` are **not** exposed to Python; Python-side choice enumeration lives in `ninjask/choices.py` (`joint_choices`, `slot_options`, `move_targets`, `team_preview_choices`).

Gotcha from testing: passing `"default"` as a choice is not a reliable shortcut (it fails at team preview, and failed mid-game in my test). Use `choices.team_preview_choices` / `choices.joint_choices` or `RandomPolicy`.

Not verified by me: the Showdown parity claims (requires Node.js 22+ and the pinned Showdown checkout), and the Rust benchmark (no Rust toolchain in the sandbox). The repo is brand new (0 stars, 0 forks, 10 commits when checked) with no public track record yet.

### 5.3 Verified Python snippets

```python
import json, ninjask
from ninjask import Battle, dex
from ninjask.policy import RandomPolicy

# Play a full game between two random policies
teams = ninjask.verification_teams()
result = ninjask.play_engine_game(
    "gen9championsvgc2026regmc", 1, teams[0], teams[1], RandomPolicy(1), RandomPolicy(2)
)
print(result.winner, result.turns)

# Drive a battle manually
d = dex("gen9championsvgc2026regmc")
assert d.team_problems(teams[0]) == []
b = Battle(d, teams[0], teams[1], seed=1)
b.set_logging(False)
policy = RandomPolicy(seed=1)
while not b.ended and b.turn < 150:
    for side in ("p1", "p2"):
        if b.needs_choice(side):
            while True:
                request = json.loads(b.request(side))
                try:
                    b.choose(side, policy.choose(request, []))  # e.g. "move 1 1, move 2 2"
                    break
                except ValueError:
                    pass  # rejected (e.g. hidden disable); request now reflects it

# Exact damage calc for the Pokémon in slots p1a -> p2a
calc = json.loads(b.damage("p1a", "expandingforce", "p2a"))
print(calc["hits"][0]["normal"], calc["critChance"])
print(b.speeds())  # {"p1a": 154, "p1b": 124, ...}
```

### 5.4 Getting Rust speed

The README documents compiling your own Rust extension into the same Python module as Ninjask's classes (depend on the crate with feature `python`, call `ninjask::python::register(m)?` in your `#[pymodule]`, and declare it under the `ninjask.extension` entry-point group). Your extension's functions can take `&PyBattle`, `&PyTracker` and `&PyDex` directly. Put observation encoding, legal joint-action enumeration and batched stepping there. "Faster batched bindings for training" is on the author's wish list.

## 6. Feasibility on a personal machine

**Compute is not the bottleneck.** Shao used one RTX 5090 for 48.3 hours. Another developer claims returns flatten after about a day on an RTX 5070 with search. MissingNo.Zero trained its Gen 1 bot on an M4 Pro laptop in ~40 hours and rented GPUs for ~$160–280 per later run. Live play runs fine on a laptop CPU.

**The simulator was the bottleneck,** and Ninjask largely removes it, provided the hot loop moves to Rust.

**Skills needed:** Python and PyTorch, a working grasp of PPO, enough Rust to write a PyO3 extension, and solid VGC mechanics knowledge. Expect months, not weeks. Jaxcalibur's author estimated a few months and several hundred dollars in GPU credits for someone with ML experience to rebuild a comparable bot; Shao had the architecture idea ~1.5 years before reaching #1.

Still to build on top of Ninjask: observation encoder, transformer policy/value network, PPO loop, league with PFSP sampling, team pool, one-turn matrix search, and a Showdown client (poke-env) for live play.

## 7. Build plan

| Phase | Goal | Done when |
|---|---|---|
| 0. Setup | Install Ninjask; run a local Showdown server; install poke-env | Random-vs-random games run in Ninjask and on the local server |
| 1. Python prototype | Observation encoder (own side + dex data), legal joint-action masks, end-to-end self-play loop with a random/heuristic policy | Thousands of games run end to end; observations and masks unit-tested |
| 2. First learning | Small policy/value transformer; slot-2-conditioned-on-slot-1 action head; PPO with terminal reward | Learned policy beats random and a simple heuristic convincingly |
| 3. Speed | Move observation encoding, joint-action enumeration and batched stepping into a Rust extension compiled with Ninjask | Throughput approaches thousands of games/s per core; GPU utilization high |
| 4. Scale training | Team pool (~1,000+ public tournament pastes, all passing `team_problems`), league with ~50% current / ~50% PFSP-weighted past snapshots | Win rate vs fixed past checkpoints keeps rising |
| 5. Search | One-turn matrix: N sampled worlds via `Tracker.world`, top-8 × top-8 joint actions, multiple RNG seeds per cell (aim 16+, ideally 64–128), value-net leaf eval, regret matching with worst-case/EV blend | Search version beats the raw network head-to-head |
| 6. Evaluation | Head-to-head vs earlier checkpoints and VGC-Bench baselines on a local server; inspect replays for odd plays | Stable, explainable improvements; known failure cases logged |

Open decisions: OTS-only first (easier, fewer unknowns) vs closed sheets; how to choose world assumptions for hidden spreads; seeds/worlds budget vs latency.

## 8. Rules and ethics

Train and test on a local Showdown server. Smogon/Showdown's initial bot policy (Oct 2026) makes a bot registry mandatory; running an unregistered bot on the public server is a bannable offense, and bot accounts should be clearly labeled. Bot authors (Jaxcalibur, MissingNo.Zero, mikumiku37) have declined to release code or weights because of cheating concerns for ladder and official Champions tournaments; consider the same before publishing weights.

## 9. References

| Resource | Link |
|---|---|
| mikumiku37 Smogon thread | https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/ |
| Ninjask | https://github.com/dcabatin/Ninjask |
| Jaxcalibur write-up | https://jaxcalibur.github.io |
| Jaxcalibur Smogon thread | https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/ |
| MissingNo.Zero | https://github.com/nickgreenquist/missingno-zero |
| PokaiTrainer paper | https://arxiv.org/abs/2608.29197 |
| VGC-Bench | https://github.com/cameronangliss/vgc-bench |
| VGC battle logs dataset | https://huggingface.co/datasets/cameronangliss/vgc-battle-logs |
| Foul Play | https://github.com/pmariglia/foul-play |
| Foul Play write-up | https://pmariglia.github.io/posts/foul-play |
| Metamon | https://metamon.tech |
| PokéChamp | https://sites.google.com/view/pokechamp-llm |
| poke-env | https://github.com/hsahovic/poke-env |
| Pokémon Showdown (simulator) | https://github.com/smogon/pokemon-showdown |
| pkmn.ai project index | https://pkmn.ai/projects/ |
| Smogon PS! Bot Projects forum | https://www.smogon.com/forums/forums/ps-bot-projects.1059/ |
| Smogon initial policy on bots | https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/ |
