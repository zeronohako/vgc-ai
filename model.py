"""Policy/value network for VGC doubles decisions (PyTorch), on `selfplay.Decisions`.

A decision becomes 13 tokens: 12 mons (own six, then the foe's six) and one global token. A mon token embeds its
species, item, ability and moves (0 = unknown, which is most of the foe's under closed sheets) and reads its
floats: the observation's 19 (HP, fainted, seen, active, status, boosts, Mega) plus the log extras (slot, turns
on the field, Protect streak, volatiles). The global token reads both sides' conditions, the field, the turn
counters and the team-preview flag. A small transformer mixes the tokens.

- Value: tanh(MLP(global token)), the expected result for the player deciding (+1 win, -1 loss).
- Policy: one logit per legal joint action (`Decisions.actions` rows), softmax within each decision. A slot's
  part is built from what it refers to: the acting mon's token, the kind (pass/move/switch/locked), the move's
  embedding, the token of its target or of the mon switched in, Mega Evolution. The two parts and the global
  token go through an MLP. A team-preview row scores its two leads and its back two.
"""

import numpy as np
import torch
from torch import nn

import selfplay as sp

MON_FLOATS = 19  # per mon in the observation (ninjask src/encode.rs)
OWN_CONDS, FOE_MONS, FOE_CONDS, FIELD = 114, 120, 234, 240  # offsets in the 250 observation floats
NULL = 12  # index of the all-zero token: a reference to nothing


def batch_from(d):
    """A Decisions (or anything with the same fields) as a dict of numpy arrays: one training/inference batch."""
    return {"ids": d.ids, "obs": d.obs, "slots": d.slots, "mon_x": d.mon_x, "glob_x": d.glob_x,
            "preview": d.preview, "pos2own": d.pos2own, "rows": d.actions, "offsets": d.offsets}


def to_torch(b, device="cpu"):
    t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in b.items()}
    for k in ("ids", "slots", "pos2own", "rows", "offsets"):
        t[k] = t[k].long()
    n = len(b["preview"])
    t["seg"] = torch.repeat_interleave(torch.arange(n, device=device), t["offsets"][1:] - t["offsets"][:-1])
    return t


class Net(nn.Module):
    def __init__(self, vocab, d=128, layers=3, heads=4):
        super().__init__()
        self.d = d
        self.species = nn.Embedding(vocab["species"] + 1, 48)
        self.item = nn.Embedding(vocab["items"] + 2, 16)  # 0 unknown, 1 known to hold nothing
        self.ability = nn.Embedding(vocab["abilities"] + 1, 16)
        self.move = nn.Embedding(vocab["moves"] + 1, 32)
        self.mon_in = nn.Linear(48 + 16 + 16 + 32 + MON_FLOATS + sp.MON_X + 3, d)
        self.side = nn.Embedding(2, d)
        self.glob_in = nn.Linear(6 + 6 + 10 + sp.GLOB_X + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.kind = nn.Embedding(4, 16)  # pass, move, switch, locked (Struggle/recharge)
        self.mega = nn.Embedding(2, 8)
        self.slot_mlp = nn.Sequential(nn.Linear(2 * d + 16 + 32 + 8, d), nn.ReLU(), nn.Linear(d, d))
        # Joint and preview scores are MLPs on concatenations, with the first layer split by input so each
        # part is computed once per distinct slot option / mon instead of once per joint row.
        self.joint_a, self.joint_b, self.joint_g = nn.Linear(d, d), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.joint_out = nn.Linear(d, 1)
        self.lead, self.back, self.preview_g = nn.Linear(d, d), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.preview_out = nn.Linear(d, 1)
        self.value_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def tokens(self, t):
        """(H (n, 13, d): 12 mon tokens + the null token, g (n, d): the global token after the encoder)."""
        ids, obs, n = t["ids"], t["obs"], len(t["preview"])
        moves = self.move(ids[..., 3:])  # (n, 2, 6, 4, 32)
        known = (ids[..., 3:] > 0).unsqueeze(-1).float()
        mean_moves = (moves * known).sum(-2) / known.sum(-2).clamp(min=1)
        floats = torch.stack([obs[:, :OWN_CONDS].reshape(n, 6, MON_FLOATS),
                              obs[:, FOE_MONS:FOE_CONDS].reshape(n, 6, MON_FLOATS)], 1)
        slot1h = nn.functional.one_hot(t["slots"].clamp(0, 2), 3).float()
        x = torch.cat([self.species(ids[..., 0]), self.item(ids[..., 1]), self.ability(ids[..., 2]), mean_moves,
                       floats, t["mon_x"], slot1h], -1)
        mons = self.mon_in(x) + self.side.weight[:, None, :]  # (n, 2, 6, d)
        glob = self.glob_in(torch.cat([obs[:, OWN_CONDS:OWN_CONDS + 6], obs[:, FOE_CONDS:FOE_CONDS + 6],
                                       obs[:, FIELD:], t["glob_x"], t["preview"].float()[:, None]], -1))
        seq = torch.cat([mons.reshape(n, 12, self.d), glob[:, None]], 1)
        pad = torch.cat([ids[..., 0].reshape(n, 12) == 0, torch.zeros(n, 1, dtype=torch.bool, device=ids.device)], 1)
        out = self.norm(self.encoder(seq, src_key_padding_mask=pad))
        out = out.masked_fill(pad[..., None], 0.0)
        H = torch.cat([out[:, :12], torch.zeros(n, 1, self.d, device=out.device)], 1)
        return H, out[:, 12]

    def forward(self, t):
        """(logits (rows,), value (n,)) for a to_torch() batch."""
        H, g = self.tokens(t)
        rows, seg, n = t["rows"], t["seg"], len(t["preview"])
        logits = torch.empty(len(rows), device=rows.device)
        pre = t["preview"][seg]
        if pre.any():  # team preview: positions in team order (own view = team order there), leads first
            r, s = rows[pre], seg[pre]
            own = H[:, :6]
            lead, back = self.lead(own), self.back(own)  # (n, 6, d) each, once per mon
            p = (r[:, :4] - 1).clamp(0, 5)
            z = lead[s, p[:, 0]] + lead[s, p[:, 1]] + back[s, p[:, 2]] + back[s, p[:, 3]] + self.preview_g(g)[s]
            logits[pre] = self.preview_out(torch.relu(z)).squeeze(-1)
        mv = ~pre
        if mv.any():
            r, s = rows[mv], seg[mv]
            slots = t["slots"]
            one_two = torch.tensor([1, 2], device=rows.device)
            # token index of the own / foe mon in slot a and b (NULL if none)
            own_hit, foe_hit = slots[:, 0, :, None] == one_two, slots[:, 1, :, None] == one_two  # (n, 6, 2)
            own_at = torch.where(own_hit.any(1), own_hit.float().argmax(1), NULL)
            foe_at = torch.where(foe_hit.any(1), 6 + foe_hit.float().argmax(1), NULL)
            own_moves = t["ids"][:, 0, :, 3:]  # (n, 6, 4)
            z = self.joint_g(g)[s]
            for i, proj in enumerate((self.joint_a, self.joint_b)):
                # the distinct options of this slot (a joint row repeats each one many times)
                f = r[:, 4 * i:4 * i + 4]
                key = (((s * 4 + f[:, 0].clamp(0, 3)) * 8 + f[:, 1].clamp(0, 7)) * 8 + f[:, 2].clamp(-2, 2) + 2) * 2 \
                    + f[:, 3].clamp(0, 1)
                _, first, inverse = unique_first(key)
                us, uf = s[first], f[first]
                kind, idx, tgt, mega = uf[:, 0], uf[:, 1], uf[:, 2], uf[:, 3]
                actor = own_at[us, i]
                move = own_moves[us, actor.clamp(max=5), (idx - 1).clamp(0, 3)]
                move = torch.where((kind == 1) & (actor < NULL), move, 0)
                ref = torch.full_like(kind, NULL)
                ref = torch.where((kind == 1) & (tgt > 0), foe_at[us, (tgt - 1).clamp(0, 1)], ref)
                ref = torch.where((kind == 1) & (tgt < 0), own_at[us, (-tgt - 1).clamp(0, 1)], ref)
                incoming = t["pos2own"][us, (idx - 1).clamp(0, 5)]
                ref = torch.where((kind == 2) & (incoming >= 0), incoming, ref)
                part = self.slot_mlp(torch.cat([H[us, actor], self.kind(kind.clamp(0, 3)), self.move(move),
                                                H[us, ref], self.mega(mega.clamp(0, 1))], -1))
                z = z + proj(part)[inverse]
            logits[mv] = self.joint_out(torch.relu(z)).squeeze(-1)
        return logits, torch.tanh(self.value_mlp(g)).squeeze(-1)


def unique_first(key):
    """(unique keys, index of each one's first occurrence, inverse) for a 1-D integer tensor."""
    u, inverse = torch.unique(key, return_inverse=True)
    first = torch.full((len(u),), len(key), dtype=torch.long, device=key.device).scatter_reduce(
        0, inverse, torch.arange(len(key), device=key.device), "amin")
    return u, first, inverse


def seg_log_softmax(logits, seg, n):
    """Log-softmax within each decision's rows (rows of decision k are contiguous)."""
    m = torch.full((n,), -torch.inf, device=logits.device).scatter_reduce(0, seg, logits.detach(), "amax")
    z = torch.zeros(n, device=logits.device).index_add(0, seg, (logits - m[seg]).exp())
    return logits - (m + z.log())[seg]


def sample(logits, seg, offsets, greedy=False):
    """One row per decision (Gumbel-max, or argmax): the choice index within the decision, as numpy."""
    n = len(offsets) - 1
    g = logits if greedy else logits - torch.log(-torch.log(torch.rand_like(logits).clamp(1e-12, 1 - 1e-7)))
    m = torch.full((n,), -torch.inf, device=g.device).scatter_reduce(0, seg, g, "amax")
    top = g == m[seg]
    first = torch.full((n,), len(g), dtype=torch.long, device=g.device).scatter_reduce(
        0, seg[top], torch.arange(len(g), device=g.device)[top], "amin")
    return (first - offsets[:-1]).cpu().numpy()
