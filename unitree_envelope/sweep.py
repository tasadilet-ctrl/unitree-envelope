#!/usr/bin/env python3
"""Sweeps Unitree's shipped MuJoCo policies: command envelope and stress tests.

Upstream ships one command -- cmd_init: [0.5, 0, 0] for every robot -- and
publishes GIFs rather than numbers. These sweeps ask what the shipped policies
actually do: which commanded velocities they can hold, how well they track
them, and how far they degrade under delay, sensor noise, payload and a
slippery floor.

    python -m unitree_envelope.sweep baseline --repo third_party/unitree_rl_gym
    python -m unitree_envelope.sweep commands --robot g1
    python -m unitree_envelope.sweep stress   --robot g1
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .rollout import ROBOTS, Stress, load_config, rollout

_CACHE: dict = {}


def _one(job):
    robot, repo, cmd, stress_kw, duration = job
    r = rollout(robot, repo, cmd=cmd, stress=Stress(**stress_kw),
                duration=duration, policy_cache=_CACHE)
    d = r.as_dict()
    d.update({f"stress_{k}": v for k, v in stress_kw.items()})
    return d


def run(jobs, workers):
    t0 = time.time()
    if workers == 1:
        rows = [_one(j) for j in jobs]
    else:
        with mp.Pool(workers) as pool:
            rows = pool.map(_one, jobs, chunksize=1)
    print(f"  {len(rows)} rollouts in {time.time() - t0:.1f}s "
          f"({workers} workers)")
    return rows


def summarise(rows, group_keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        out.setdefault(key, []).append(row)
    table = []
    for key, rs in sorted(out.items(), key=lambda kv: str(kv[0])):
        table.append(dict(
            **dict(zip(group_keys, key)),
            n=len(rs),
            survival=float(np.mean([r["survived"] for r in rs])),
            mean_vx=float(np.mean([r["mean_vx"] for r in rs])),
            err_vx=float(np.mean([abs(r["track_err_vx"]) for r in rs])),
            err_vy=float(np.mean([abs(r["track_err_vy"]) for r in rs])),
            err_yaw=float(np.mean([abs(r["track_err_yaw"]) for r in rs])),
            min_h=float(np.mean([r["min_height_ratio"] for r in rs])),
        ))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("baseline", "commands", "stress"))
    ap.add_argument("--repo", default="third_party/unitree_rl_gym")
    ap.add_argument("--robot", default=None, help="default: all three")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jitter", type=float, default=0.02,
                    help="per-joint rad jitter at reset; the seeds differ only in this")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    repo = str(Path(args.repo).resolve())
    robots = [args.robot] if args.robot else list(ROBOTS)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    def st(**kw):
        return dict(init_qpos_noise=args.jitter, **kw)

    if args.mode == "baseline":
        jobs, shipped = [], {}
        for rb in robots:
            shipped[rb] = load_config(rb, repo)["cmd_init"]
            for s in seeds:
                jobs.append((rb, repo, None, st(seed=s), args.duration))
        rows = run(jobs, args.workers)
        table = summarise(rows, ["robot"])
        print(f"\nshipped command per robot: "
              f"{ {k: v for k, v in shipped.items()} }")
        print(f"{'robot':8}{'n':>3}{'survival':>10}{'cmd vx':>8}{'mean vx':>9}"
              f"{'|err| vx':>10}{'min h':>8}")
        for t in table:
            print(f"{t['robot']:8}{t['n']:>3}{t['survival']:>9.0%}"
                  f"{shipped[t['robot']][0]:>8.2f}{t['mean_vx']:>9.3f}"
                  f"{t['err_vx']:>10.3f}{t['min_h']:>8.3f}")
        out = dict(mode="baseline", duration=args.duration, seeds=args.seeds,
                   jitter=args.jitter, shipped_cmd=shipped, table=table, rows=rows)

    elif args.mode == "commands":
        vxs = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
        vys = [-0.5, -0.25, 0.0, 0.25, 0.5]
        yaws = [-1.0, -0.5, 0.0, 0.5, 1.0]
        jobs = []
        for rb in robots:
            for vx, vy in itertools.product(vxs, vys):
                for s in seeds:
                    jobs.append((rb, repo, [vx, vy, 0.0], st(seed=s), args.duration))
            for yaw in yaws:
                if yaw == 0.0:
                    continue
                for s in seeds:
                    jobs.append((rb, repo, [0.5, 0.0, yaw], st(seed=s), args.duration))
        rows = run(jobs, args.workers)
        table = summarise(rows, ["robot", "cmd"])
        for rb in robots:
            print(f"\n{rb}: survival %, rows = commanded vx, cols = vy")
            print("        " + "".join(f"{v:>8.2f}" for v in vys))
            for vx in vxs:
                cells = []
                for vy in vys:
                    hit = [t for t in table if t["robot"] == rb
                           and tuple(t["cmd"]) == (vx, vy, 0.0)]
                    cells.append(f"{hit[0]['survival']:>7.0%}" if hit else "      -")
                print(f"{vx:>7.2f} " + "".join(cells))
        out = dict(mode="commands", duration=args.duration, seeds=args.seeds,
                   jitter=args.jitter, vxs=vxs, vys=vys, yaws=yaws,
                   table=table, rows=rows)

    else:
        axes = {
            "obs_latency_steps": [0, 1, 2, 3, 4, 6, 8],      # control steps @50 Hz
            "obs_noise_std": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
            "added_trunk_mass": [0.0, 2.0, 5.0, 10.0, 15.0, 20.0],
            "friction_scale": [1.0, 0.7, 0.5, 0.3, 0.2, 0.1],
        }
        jobs = []
        for rb in robots:
            for axis, values in axes.items():
                for v in values:
                    for s in seeds:
                        jobs.append((rb, repo, None, st(seed=s, **{axis: v}),
                                     args.duration))
        rows = run(jobs, args.workers)
        print()
        for rb in robots:
            for axis, values in axes.items():
                sub = [r for r in rows if r["robot"] == rb
                       and r.get(f"stress_{axis}") is not None]
                line = []
                for v in values:
                    rs = [r for r in sub if r[f"stress_{axis}"] == v]
                    if rs:
                        line.append(f"{v:g}:{np.mean([r['survived'] for r in rs]):.0%}")
                print(f"  {rb:6} {axis:20} " + "  ".join(line))
        out = dict(mode="stress", duration=args.duration, seeds=args.seeds,
                   jitter=args.jitter, axes=axes, rows=rows)

    path = Path(args.out) / f"{args.mode}.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
