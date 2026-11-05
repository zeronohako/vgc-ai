"""Download public Reg M-C Bo3 replays from Showdown and build a team pool from their team sheets.

Usage: python fetch_regmc.py [--max 6000] [--delay 1.0]      # fetch up to --max new replays, then build the pool
       python fetch_regmc.py --max 0                          # only rebuild the pool from the logs on disk

Every Bo3 log has both teams as `|showteam|` lines: species, item, ability, moves and nature, but no stat
points. Resumable: paging always goes back from the newest replay and skips replays already in LOGS_OUT
(re-reading a page of known ones costs one search request). One request per `--delay` seconds.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

import ninjask

BASE = "https://replay.pokemonshowdown.com"
SEARCH_FORMAT = "gen9championsvgc2026regmcbo3"
DEX_FORMAT = "gen9championsvgc2026regmcots"  # Bo3 games are played with open sheets
LOGS_OUT = "data/logs_regmc.json"  # replay_id -> [uploadtime, log_text, rating]  (logs_regmb.json's shape + rating)
POOL_OUT = "data/teams_regmc.txt"  # one packed team per line, best rating first
META_OUT = "data/teams_regmc_meta.json"  # per pool line: best rating, replays seen in, newest upload, a few ids
USER_AGENT = "vgc-ai-research (replay scraper)"


def get_json(url, delay, tries=6):
    """GET url as JSON, sleeping `delay` first; None on 404. Retries other failures with backoff."""
    for k in range(tries):
        time.sleep(delay if k == 0 else delay * 2 ** k)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            err = e
        print(f"  retry {k + 1}/{tries - 1} after {err!r}: {url}", flush=True)
    raise RuntimeError(f"giving up on {url}")


def save(logs, path=LOGS_OUT):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(logs, f)
    os.replace(tmp, path)  # never leave a half-written file behind


def fetch(logs, max_new, delay):
    """Add up to max_new replays to `logs` (in place). Returns how many were added."""
    added = 0
    before = None  # None = newest first
    while added < max_new:
        url = f"{BASE}/search.json?format={SEARCH_FORMAT}" + (f"&before={before}" if before else "")
        page = get_json(url, delay) or []
        fresh = [r for r in page if r["id"] not in logs and not r.get("private") and not r.get("password")]
        for r in fresh:
            if added >= max_new:
                break
            d = get_json(f"{BASE}/{r['id']}.json", delay)
            if not d or "|showteam|" not in d.get("log", ""):
                continue
            logs[r["id"]] = [d["uploadtime"], d["log"], r.get("rating") or d.get("rating")]  # .json's is often null
            added += 1
            if added % 50 == 0:
                print(f"  {added} new replays (newest page entry {time.strftime('%Y-%m-%d %H:%M', time.gmtime(r['uploadtime']))} UTC)", flush=True)
            if added % 200 == 0:
                save(logs)
        if len(page) < 51:  # the search returns 51 when there are more
            print("  reached the end of the search", flush=True)
            break
        # +1 plus the id check: replays sharing the boundary timestamp are not lost
        before = page[-1]["uploadtime"] + 1
    save(logs)
    return added


def mon_key(mon):
    """A set without its order of moves: species, item, ability, sorted moves, nature."""
    f = mon.split("|")
    return (f[0] or f[1], f[2], f[3], tuple(sorted(f[4].split(","))), f[5])


def player_ratings(lines):
    """{'p1': 1245, ...} from `|player|p1|name|avatar|rating` lines (the search's replay rating is often null)."""
    out = {}
    for l in lines:
        w = l.split("|")
        if w[1:2] == ["player"] and len(w) > 5 and w[5].isdigit():
            out.setdefault(w[2], int(w[5]))
    return out


def build_pool(logs):
    """Unique teams that Ninjask simulates fully, best rating of a player who used it first. Returns (teams, meta)."""
    dx = ninjask.dex(DEX_FORMAT)
    seen, problems = {}, 0
    for rid, (uploaded, log, _) in logs.items():
        lines = log.split("\n")
        ratings = player_ratings(lines)
        for line in lines:
            if not line.startswith("|showteam|"):
                continue
            side, team = line.split("|", 3)[2:]
            rating = ratings.get(side)
            key = tuple(sorted(mon_key(m) for m in team.split("]")))
            if key not in seen:
                if dx.team_problems(team):
                    problems += 1
                    seen[key] = None
                    continue
                seen[key] = dict(team=team, rating=rating or 0, seen=0, last=0, replays=[])
            e = seen[key]
            if e is None:
                continue
            e["rating"] = max(e["rating"], rating or 0)
            e["seen"] += 1
            e["last"] = max(e["last"], uploaded)
            if len(e["replays"]) < 5:
                e["replays"].append(rid)
    pool = sorted((e for e in seen.values() if e), key=lambda e: (-e["rating"], -e["seen"]))
    print(f"pool: {len(pool)} unique teams from {len(logs)} replays ({problems} unique teams rejected by team_problems)")
    return [e["team"] for e in pool], [{k: e[k] for k in ("rating", "seen", "last", "replays")} for e in pool]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=6000, help="new replays to fetch this run (0: only rebuild the pool)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    logs = json.load(open(LOGS_OUT, encoding="utf-8")) if os.path.exists(LOGS_OUT) else {}
    print(f"{len(logs)} replays on disk")
    if args.max > 0:
        print(f"added {fetch(logs, args.max, args.delay)} replays, {len(logs)} total")
    teams, meta = build_pool(logs)
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(teams) + "\n")
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
