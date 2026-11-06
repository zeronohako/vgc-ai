"""PPO self-play on the Reg M-C team pool, closed team sheets.

Usage: python train.py [--minutes 30] [--run NAME] [--battles 256] [--steps 32] [--d 64 --layers 2] ...

One network (model.Net) plays both sides of `--battles` battles between random pool teams (selfplay.SelfPlay).
Every `--steps` env steps the decisions collected train it with PPO: reward +1 win / -1 loss / 0 draw at the end
of a game, nothing before (GAE over each player's own decisions). Every `--eval-every` iterations it plays a
uniformly random player and logs its win rate. Checkpoint + log in runs/<NAME>/ (resumes if present, so a cloud
session that restarts every 4 h just runs the same command again).
"""

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # torch's threads mustn't spin against the engine's

import argparse
import json
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
    """(choices, logp of each choice, value) for every decision of an observe()."""
    t = M.to_torch(M.batch_from(d))
    with torch.no_grad():
        logits, value = net(t)
        lp = M.seg_log_softmax(logits, t["seg"], d.n)
        choice = M.sample(logits, t["seg"], t["offsets"], greedy)
    return choice, lp[t["offsets"][:-1] + torch.from_numpy(choice)].numpy(), value.numpy()


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
    rows = np.concatenate([np.arange(d.offsets[i], d.offsets[i + 1]) for i in k])
    return sp.Decisions(battle=d.battle[k], side=d.side[k], preview=d.preview[k], obs=d.obs[k], ids=d.ids[k],
                        actions=d.actions[rows], offsets=np.r_[0, np.cumsum(c)], slots=d.slots[k],
                        pos2own=d.pos2own[k], mon_x=d.mon_x[k], glob_x=d.glob_x[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cpu-smoke")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--battles", type=int, default=256)
    ap.add_argument("--steps", type=int, default=32, help="env steps per iteration")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=256)
    ap.add_argument("--threads", type=int, default=os.cpu_count() // 2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = os.path.join("runs", args.run)
    os.makedirs(run, exist_ok=True)
    ckpt_path, log_path = os.path.join(run, "ckpt.pt"), os.path.join(run, "log.jsonl")

    teams = sp.load_pool()
    env = sp.SelfPlay(teams, args.battles, seed=args.seed)
    net = M.Net(env.dex.vocab(), d=args.d, layers=args.layers, heads=args.heads)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    it, trained_s, decisions_total, games_total = 0, 0.0, 0, 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        it, trained_s, decisions_total, games_total = ck["it"], ck["seconds"], ck["decisions"], ck["games"]
        print(f"resumed {ckpt_path} at iteration {it} ({trained_s / 60:.1f} min trained)")
    print(f"{sum(p.numel() for p in net.parameters())} parameters; {len(teams)} pool teams; {args.battles} battles")
    if it == 0:
        wr = evaluate(net, teams, args.eval_games)
        print(f"untrained vs random: {wr:.1%}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"it": 0, "decisions": 0, "games": 0, "minutes": 0, "vs_random": wr}) + "\n")

    start = time.time()
    while time.time() - start < args.minutes * 60:
        t0 = time.time()
        ro = Rollout()
        net.eval()
        for _ in range(args.steps):
            d = env.observe()
            choice, logp, value = act(net, d)
            ended = env.step(choice)
            ro.add(d, choice, logp, value, env.refused, ended)
        games = ro.finish(args.gamma, args.lam) // 2
        t1 = time.time()

        net.train()
        stats = []
        for _ in range(args.epochs):
            for mb in np.array_split(rng.permutation(ro.valid_idx), max(1, len(ro.valid_idx) // args.minibatch)):
                t = M.to_torch(ro.batch(mb))
                logits, v = net(t)
                n = len(mb)
                lp_all = M.seg_log_softmax(logits, t["seg"], n)
                lp = lp_all[t["offsets"][:-1] + torch.from_numpy(ro.choice[mb])]
                ratio = torch.exp(lp - torch.from_numpy(ro.logp[mb]))
                a = torch.from_numpy(ro.adv[mb])
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vl = ((v - torch.from_numpy(ro.ret[mb])) ** 2).mean()
                ent = -torch.zeros(n).index_add(0, t["seg"], lp_all.exp() * lp_all).mean()
                loss = pg + args.vf * vl - args.ent * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kl = (torch.from_numpy(ro.logp[mb]) - lp).mean().item()
                    clipped = ((ratio - 1).abs() > args.clip).float().mean().item()
                stats.append((pg.item(), vl.item(), ent.item(), kl, clipped))
        t2 = time.time()

        it += 1
        trained_s += t2 - t0
        decisions_total += ro.n
        games_total += games
        pg, vl, ent, kl, clipped = np.mean(stats, axis=0)
        rec = {"it": it, "decisions": decisions_total, "games": games_total, "minutes": round(trained_s / 60, 2),
               "pg": round(pg, 4), "vloss": round(vl, 4), "entropy": round(ent, 3), "kl": round(kl, 4),
               "clipfrac": round(clipped, 3), "collect_s": round(t1 - t0, 1), "train_s": round(t2 - t1, 1)}
        if it % args.eval_every == 0:
            rec["vs_random"] = evaluate(net, teams, args.eval_games)
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        tmp = ckpt_path + ".tmp"
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it, "seconds": trained_s,
                    "decisions": decisions_total, "games": games_total, "args": vars(args)}, tmp)
        os.replace(tmp, ckpt_path)


if __name__ == "__main__":
    main()
