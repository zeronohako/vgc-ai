"""PPO self-play on the Reg M-C team pool, closed team sheets.

Usage: python train.py [--minutes 30] [--run NAME] [--battles 256] [--steps 32] [--d 64 --layers 2]
                       [--pfsp 0.5] [--target-kl 0.01] ...

One network (model.Net) plays both sides of `--battles` battles between random pool teams (selfplay.SelfPlay).
Every `--steps` env steps the decisions collected train it with PPO: reward +1 win / -1 loss / 0 draw at the end
of a game, nothing before (GAE over each player's own decisions). Every `--eval-every` iterations it plays a
uniformly random player and logs its win rate. Checkpoint + log in <runs-dir>/<NAME>/ (resumes if present, so a
cloud session that restarts just runs the same command again), plus a snapshot of the network every
`--snapshot-every` iterations (snap_<iteration>.pt, CPU tensors) for arena.py to play against.

PFSP (--pfsp F): in a fraction F of the battles one side is played by a past snapshot of this run instead (see
League); only the learner's decisions there are trained on. --target-kl adapts the learning rate after every
iteration so the mean KL per update stays near it (--lr is then the ceiling); without it the rate is --lr, also
on resume.

GPU: --device cuda (the default when one is there) trains on it; the battles always run on the CPU. Moves during
the rollouts are picked on --act-device (default: the same device); `--act-device cpu` keeps a CPU copy of the
network for that, synced after every update, in case many small GPU calls are slow. --resume-from DIR copies a
previous session's run folder (ckpt.pt, log.jsonl, snapshots) into the run folder first, if it has no checkpoint
yet: a Kaggle session reads the last one's output from /kaggle/input and writes its own to /kaggle/working.
"""

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # torch's threads mustn't spin against the engine's

import argparse
import json
import shutil
import time
import warnings

import numpy as np
import torch

import model as M
import selfplay as sp

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
FIELDS = ("ids", "obs", "slots", "mon_x", "glob_x", "preview", "pos2own")


class Rollout:
    """Decisions of one iteration, linked into each player's trajectory, ready for GAE and minibatches."""

    def __init__(self):
        self.steps, self.choice, self.logp, self.value = [], [], [], []
        self.next, self.reward, self.terminal, self.valid = [], [], [], []
        self.last = {}  # (battle, side) -> index of that player's last applied decision
        self.n = 0

    def add(self, d, choice, logp, value, refused, ended):
        base = self.n
        self.steps.append({**{f: getattr(d, f) for f in FIELDS}, "rows": d.actions, "counts": d.counts()})
        self.choice.append(choice)
        self.logp.append(logp)
        self.value.append(value)
        self.next += [-1] * d.n
        self.reward += [0.0] * d.n
        self.terminal += [False] * d.n
        self.valid += (~refused).tolist()  # a refused choice never happened: it is asked again
        for k in np.flatnonzero(~refused):
            key = (int(d.battle[k]), int(d.side[k]))
            if key in self.last:
                self.next[self.last[key]] = base + int(k)
            self.last[key] = base + int(k)
        for slot, winner, _ in ended:
            for side in (0, 1):
                g = self.last.pop((slot, side), None)
                if g is not None:
                    self.terminal[g] = True
                    self.reward[g] = 0.0 if winner is None else (1.0 if winner == side else -1.0)
        self.n += d.n

    def finish(self, gamma, lam):
        """Concatenate, then advantages and returns (a trajectory cut by the iteration end bootstraps with 0)."""
        cat = {k: np.concatenate([s[k] for s in self.steps]) for k in self.steps[0]}
        self.data = cat
        self.row_start = np.r_[0, np.cumsum(cat["counts"])[:-1]]
        self.choice, self.logp, self.value = map(np.concatenate, (self.choice, self.logp, self.value))
        nxt, rew, term, valid = (np.array(x) for x in (self.next, self.reward, self.terminal, self.valid))
        adv = np.zeros(self.n, np.float32)
        for g in range(self.n - 1, -1, -1):
            if not valid[g]:
                continue
            if term[g]:
                adv[g] = rew[g] - self.value[g]
            elif nxt[g] >= 0:
                adv[g] = gamma * self.value[nxt[g]] - self.value[g] + gamma * lam * adv[nxt[g]]
        self.adv, self.ret, self.valid_idx = adv, adv + self.value, np.flatnonzero(valid)
        return int(term.sum())

    def batch(self, idx):
        """Minibatch of decisions idx, in model.to_torch's layout."""
        c = self.data["counts"][idx]
        starts = self.row_start[idx]
        rows = np.repeat(starts - np.r_[0, np.cumsum(c)[:-1]], c) + np.arange(c.sum())
        b = {f: self.data[f][idx] for f in FIELDS}
        b["rows"], b["offsets"] = self.data["rows"][rows], np.r_[0, np.cumsum(c)]
        return b


def act(net, d, greedy=False):
    """(choices, logp of each choice, value) for every decision of an observe(), on the net's device."""
    dev = next(net.parameters()).device
    t = M.to_torch(M.batch_from(d), dev)
    with torch.no_grad():
        logits, value = net(t)
        lp = M.seg_log_softmax(logits, t["seg"], d.n)
        choice = M.sample(logits, t["seg"], t["offsets"], greedy)
        picked = lp[t["offsets"][:-1] + torch.from_numpy(choice).to(dev)]
    return choice, picked.cpu().numpy(), value.cpu().numpy()


def evaluate(net, teams, games, battles=64, seed=12345):
    """Win rate of `net` (sampling) against a uniformly random player; the net is p1 in even battles, p2 in odd."""
    env = sp.SelfPlay(teams, battles, seed=seed)
    rng = np.random.default_rng(seed)
    score = played = 0.0
    while played < games:
        d = env.observe()
        choice = (rng.random(d.n) * d.counts()).astype(np.int64)
        mine = d.side == d.battle % 2
        if mine.any():
            sub = np.flatnonzero(mine)
            c, _, _ = act(net, subset(d, sub))
            choice[sub] = c
        for slot, winner, _ in env.step(choice):
            played += 1
            score += 0.5 if winner is None else float(winner == slot % 2)
    return score / played


def subset(d, k):
    """The decisions k of a Decisions (with their action rows)."""
    c = d.counts()[k]
    rows = np.repeat(d.offsets[k] - np.r_[0, np.cumsum(c)[:-1]], c) + np.arange(c.sum())
    return sp.Decisions(battle=d.battle[k], side=d.side[k], preview=d.preview[k], obs=d.obs[k], ids=d.ids[k],
                        actions=d.actions[rows], offsets=np.r_[0, np.cumsum(c)], slots=d.slots[k],
                        pos2own=d.pos2own[k], mon_x=d.mon_x[k], glob_x=d.glob_x[k])


def load_net(path, vocab, device="cpu"):
    """A checkpoint or snapshot as an eval-mode Net on `device` (its size comes from the args saved with it)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    net = M.Net(vocab, d=a["d"], layers=a["layers"], heads=a["heads"]).to(device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net


class League:
    """Prioritized fictitious self-play (AlphaStar's "hard" PFSP) against this run's own snapshots.

    In the first round(frac * battles) battle slots one side (random per game) is the learner and the other a past
    snapshot. Every iteration, refresh() draws up to k snapshots, each with weight (1 - p)^2, p = the learner's
    recent score against it (an exponential average over its games, 0.5 before any): the ones it struggles with come
    up most, the ones it beats easily hardly ever. A league slot keeps its opponent for the whole game and draws a
    new one (and a new side) from the current draw when the game ends. Until the first snapshot exists, league
    slots are plain self-play.
    """

    ALPHA = 0.02  # weight of one game in the score average (~50 games of memory)

    def __init__(self, run, frac, battles, k, vocab, device, rng, state=None):
        self.run, self.k, self.vocab, self.device, self.rng = run, k, vocab, device, rng
        self.n = int(round(frac * battles))
        self.opp = np.full(battles, -1, np.int64)  # per slot: index into self.files, -1 = self-play
        self.side = rng.integers(2, size=battles)  # per slot: the learner's side
        self.files, self.fid, self.nets, self.active = [], {}, {}, []
        self.p = dict((state or {}).get("p", {}))  # snapshot file -> learner's score against it
        self.started = False
        self.score = self.games = 0.0

    def _id(self, f):
        if f not in self.fid:
            self.fid[f] = len(self.files)
            self.files.append(f)
        return self.fid[f]

    def _draw(self, slot):
        self.opp[slot] = self._id(self.active[self.rng.integers(len(self.active))])
        self.side[slot] = self.rng.integers(2)

    def refresh(self):
        """New draw of opponents for this iteration (call before its first observe)."""
        snaps = sorted(f for f in os.listdir(self.run) if f.startswith("snap_") and f.endswith(".pt"))
        if snaps:
            w = np.array([(1 - self.p.get(f, 0.5)) ** 2 for f in snaps]) + 1e-4
            pick = self.rng.choice(len(snaps), size=min(self.k, len(snaps)), replace=False, p=w / w.sum())
            self.active = [snaps[i] for i in pick]
            if not self.started:  # the env was just created: every game is at its start
                for slot in range(self.n):
                    self._draw(slot)
        self.started = True
        keep = set(self.active) | {self.files[j] for j in set(self.opp.tolist()) if j >= 0}
        if len(self.nets) > 4 * self.k:
            for f in [f for f in self.nets if f not in keep]:
                del self.nets[f]

    def net(self, f):
        if f not in self.nets:
            self.nets[f] = load_net(os.path.join(self.run, f), self.vocab, self.device)
        return self.nets[f]

    def split(self, d):
        """(mask of the learner's decisions, {snapshot file: indices of the decisions it plays})."""
        o = self.opp[d.battle]
        theirs = (o >= 0) & (d.side != self.side[d.battle])
        return ~theirs, {self.files[j]: np.flatnonzero(theirs & (o == j)) for j in np.unique(o[theirs]).tolist()}

    def ended(self, ended):
        """Score the league games that ended (env.step's list) and give their slots new opponents."""
        for slot, winner, _ in ended:
            j = self.opp[slot]
            if j >= 0:
                f = self.files[j]
                s = 0.5 if winner is None else float(winner == self.side[slot])
                p = self.p.get(f, 0.5)
                self.p[f] = p + self.ALPHA * (s - p)
                self.score += s
                self.games += 1
            if slot < self.n and self.active:
                self._draw(slot)

    def stats(self):
        """Log fields for the iteration (and a reset): score vs the league, games, the iterations drawn."""
        rec = {"league": round(self.score / self.games, 3) if self.games else None, "league_games": int(self.games),
               "opp": [int(f[5:-3]) for f in self.active]}
        self.score = self.games = 0.0
        return rec

    def hardest(self, n=3):
        """The n snapshots with the lowest learner score so far, as [iteration, score]."""
        return [[int(f[5:-3]), round(p, 3)] for f, p in sorted(self.p.items(), key=lambda x: x[1])[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cpu-smoke")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--resume-from", help="a previous session's run folder, copied in if this run has no checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--act-device", help="device for picking moves in the rollouts (default: --device)")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--battles", type=int, default=256)
    ap.add_argument("--steps", type=int, default=32, help="env steps per iteration")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate (with --target-kl: its ceiling)")
    ap.add_argument("--target-kl", type=float, default=0,
                    help="adapt the learning rate so the mean KL per update stays near this (0: constant --lr)")
    ap.add_argument("--lr-min", type=float, default=1e-5, help="floor of the adapted learning rate")
    ap.add_argument("--pfsp", type=float, default=0,
                    help="fraction of the battles where one side is a past snapshot of this run (0: pure self-play)")
    ap.add_argument("--pfsp-opponents", type=int, default=4, help="snapshots drawn per iteration")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=256)
    ap.add_argument("--snapshot-every", type=int, default=10, help="iterations between snapshots (0: none)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() // 2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = os.path.join(args.runs_dir, args.run)
    os.makedirs(run, exist_ok=True)
    ckpt_path, log_path = os.path.join(run, "ckpt.pt"), os.path.join(run, "log.jsonl")
    if args.resume_from and not os.path.exists(ckpt_path):
        for f in sorted(os.listdir(args.resume_from)):
            if f in ("ckpt.pt", "log.jsonl") or (f.startswith("snap_") and f.endswith(".pt")):
                shutil.copy2(os.path.join(args.resume_from, f), run)
        print(f"copied {args.resume_from} into {run}")
    dev, act_dev = torch.device(args.device), torch.device(args.act_device or args.device)

    teams = sp.load_pool()
    env = sp.SelfPlay(teams, args.battles, seed=args.seed)
    net = M.Net(env.dex.vocab(), d=args.d, layers=args.layers, heads=args.heads).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    it, trained_s, decisions_total, games_total, ck = 0, 0.0, 0, 0, {}
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])  # this also restores the saved learning rate
        it, trained_s, decisions_total, games_total = ck["it"], ck["seconds"], ck["decisions"], ck["games"]
        print(f"resumed {ckpt_path} at iteration {it} ({trained_s / 60:.1f} min trained)")
    lr = float(np.clip(opt.param_groups[0]["lr"], args.lr_min, args.lr)) if args.target_kl and ck else args.lr
    for g in opt.param_groups:
        g["lr"] = lr
    actor = net
    if act_dev != dev:
        actor = M.Net(env.dex.vocab(), d=args.d, layers=args.layers, heads=args.heads).to(act_dev)
        actor.load_state_dict(net.state_dict())
        actor.eval()
    gpu_name = f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""
    league = None
    if args.pfsp:
        league = League(run, args.pfsp, args.battles, args.pfsp_opponents, env.dex.vocab(), act_dev, rng,
                        state=ck.get("pfsp"))
    print(f"{sum(p.numel() for p in net.parameters())} parameters; {len(teams)} pool teams; {args.battles} battles; "
          f"training on {dev}{gpu_name}, moves on {act_dev}; lr {lr:.3g}"
          + (f", target KL {args.target_kl}" if args.target_kl else "")
          + (f"; PFSP in {league.n} battles, {args.pfsp_opponents} snapshots per iteration" if league else ""))
    if it == 0:
        wr = evaluate(actor, teams, args.eval_games)
        print(f"untrained vs random: {wr:.1%}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"it": 0, "decisions": 0, "games": 0, "minutes": 0, "vs_random": wr}) + "\n")

    def on_dev(x):
        return torch.from_numpy(x).to(dev)

    start = time.time()
    while time.time() - start < args.minutes * 60:
        t0 = time.time()
        ro = Rollout()
        net.eval()
        games = played = 0
        if league:
            league.refresh()
        for _ in range(args.steps):
            d = env.observe()
            if league is None:
                choice, logp, value = act(actor, d)
                ended = env.step(choice)
                ro.add(d, choice, logp, value, env.refused, ended)
            else:  # the learner's decisions are trained on, the snapshots' only played
                mine, theirs = league.split(d)
                dl = d if mine.all() else subset(d, np.flatnonzero(mine))
                choice = np.zeros(d.n, np.int64)
                if dl.n:
                    c, logp, value = act(actor, dl)
                    choice[mine] = c
                else:
                    c, logp, value = np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32)
                for f, k in theirs.items():
                    choice[k] = act(league.net(f), subset(d, k))[0]
                ended = env.step(choice)
                ro.add(dl, c, logp, value, env.refused[mine], ended)  # always: it scores the games that ended
                league.ended(ended)
            games += len(ended)
            played += d.n
        ro.finish(args.gamma, args.lam)
        t1 = time.time()

        net.train()
        stats = []
        for _ in range(args.epochs):
            for mb in np.array_split(rng.permutation(ro.valid_idx), max(1, len(ro.valid_idx) // args.minibatch)):
                t = M.to_torch(ro.batch(mb), dev)
                logits, v = net(t)
                n = len(mb)
                lp_all = M.seg_log_softmax(logits, t["seg"], n)
                lp = lp_all[t["offsets"][:-1] + on_dev(ro.choice[mb])]
                ratio = torch.exp(lp - on_dev(ro.logp[mb]))
                a = on_dev(ro.adv[mb])
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vl = ((v - on_dev(ro.ret[mb])) ** 2).mean()
                ent = -torch.zeros(n, device=dev).index_add(0, t["seg"], lp_all.exp() * lp_all).mean()
                loss = pg + args.vf * vl - args.ent * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kl = (on_dev(ro.logp[mb]) - lp).mean().item()
                    clipped = ((ratio - 1).abs() > args.clip).float().mean().item()
                stats.append((pg.item(), vl.item(), ent.item(), kl, clipped))
        if actor is not net:
            actor.load_state_dict(net.state_dict())
        t2 = time.time()

        it += 1
        trained_s += t2 - t0
        decisions_total += ro.n
        games_total += games
        pg, vl, ent, kl, clipped = np.mean(stats, axis=0)
        rec = {"it": it, "decisions": decisions_total, "games": games_total, "minutes": round(trained_s / 60, 2),
               "pg": round(pg, 4), "vloss": round(vl, 4), "entropy": round(ent, 3), "kl": round(kl, 4),
               "clipfrac": round(clipped, 3), "collect_s": round(t1 - t0, 1), "train_s": round(t2 - t1, 1),
               "per_s": round(ro.n / (t2 - t0)), "lr": float(f"{lr:.3g}")}
        if league:
            rec.update(league.stats(), played=played)
        if args.target_kl:  # KL grows ~ with lr^2: aim at the target, at most x1.5 or x2/3 per iteration
            lr = float(np.clip(lr * np.clip(np.sqrt(args.target_kl / max(kl, 1e-8)), 2 / 3, 1.5),
                               args.lr_min, args.lr))
            for g in opt.param_groups:
                g["lr"] = lr
        if it % args.eval_every == 0:
            rec["vs_random"] = evaluate(actor, teams, args.eval_games)
            if league:
                rec["hardest"] = league.hardest()
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        tmp = ckpt_path + ".tmp"
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it, "seconds": trained_s,
                    "decisions": decisions_total, "games": games_total, "args": vars(args),
                    "pfsp": {"p": league.p} if league else ck.get("pfsp")}, tmp)
        os.replace(tmp, ckpt_path)
        if args.snapshot_every and it % args.snapshot_every == 0:
            torch.save({"net": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "it": it, "minutes": rec["minutes"], "decisions": decisions_total,
                        "games": games_total, "args": vars(args)}, os.path.join(run, f"snap_{it:05d}.pt"))


if __name__ == "__main__":
    main()
