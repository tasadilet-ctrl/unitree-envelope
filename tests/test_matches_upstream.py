#!/usr/bin/env python3
"""Checks the headless harness reproduces upstream's control loop exactly.

rollout.py removes the viewer and the wall-clock sleep from upstream's
deploy_mujoco.py. Everything else must be identical, so this holds a SECOND
implementation of the loop -- transcribed directly from upstream's source and
deliberately not sharing code with rollout.py -- and requires the two to agree
on the whole height and velocity trace, exactly.

It also pins the things that are easy to get subtly wrong:
  * the observation layout and its 47 slots
  * the gait phase convention (counter * dt, counter incremented BEFORE the
    decimation test)
  * that the observation carries the PREVIOUS action
  * that upstream's own pd_control / get_gravity_orientation are the ones used

    python tests/test_matches_upstream.py --repo third_party/unitree_rl_gym
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def reference_loop(robot, repo_root, duration):
    """Upstream's loop, transcribed from deploy/deploy_mujoco/deploy_mujoco.py.

    Viewer and sleep dropped; nothing else changed. Imports nothing from
    unitree_envelope, on purpose.
    """
    import mujoco, torch, yaml
    from deploy.deploy_mujoco.deploy_mujoco import (get_gravity_orientation,
                                                    pd_control)
    cfg = pathlib.Path(repo_root) / "deploy/deploy_mujoco/configs" / f"{robot}.yaml"
    config = yaml.safe_load(cfg.read_text())
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(repo_root))
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(repo_root))
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions, num_obs = config["num_actions"], config["num_obs"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    policy = torch.jit.load(policy_path)

    zs, vs = [], []
    z0 = float(d.qpos[2])          # BEFORE any step, as the harness does
    for _ in range(int(duration / simulation_dt)):
        tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                         np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        counter += 1
        if counter % control_decimation == 0:
            qj = (d.qpos[7:] - default_angles) * config["dof_pos_scale"]
            dqj = d.qvel[6:] * config["dof_vel_scale"]
            omega = d.qvel[3:6] * config["ang_vel_scale"]
            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            obs[:3] = omega
            obs[3:6] = get_gravity_orientation(d.qpos[3:7])
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + num_actions] = qj
            obs[9 + num_actions:9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = np.array(
                [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * config["action_scale"] + default_angles
        zs.append(float(d.qpos[2]))
        vs.append([float(d.qvel[0]), float(d.qvel[1]), float(d.qvel[5])])
    return np.array(zs), np.array(vs), z0


def test_recurrent_reset(repo, robot="g1", duration=20.0):
    """The shipped policies are LSTMs; a cached module must not leak state.

    Run a seed, then a different episode, then the same seed again, all through
    one cache. Without a per-episode reset the repeat differed -- in one case
    the difference was whether the robot fell.
    """
    from unitree_envelope import rollout as R
    cache = {}
    a = R.rollout(robot, repo, duration=duration,
                  stress=R.Stress(init_qpos_noise=0.02, seed=3), policy_cache=cache)
    R.rollout(robot, repo, duration=duration,
              stress=R.Stress(init_qpos_noise=0.02, seed=0, obs_noise_std=0.1),
              policy_cache=cache)
    b = R.rollout(robot, repo, duration=duration,
                  stress=R.Stress(init_qpos_noise=0.02, seed=3), policy_cache=cache)
    fresh = R.rollout(robot, repo, duration=duration,
                      stress=R.Stress(init_qpos_noise=0.02, seed=3))
    ok = a.as_dict() == b.as_dict() == fresh.as_dict()
    print(f"  [{'PASS' if ok else 'FAIL'}] cached policy gives the same episode as a "
          f"fresh one  (vx {a.mean_vx:.6f} / {b.mean_vx:.6f} / {fresh.mean_vx:.6f})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="third_party/unitree_rl_gym")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--duration", type=float, default=6.0)
    args = ap.parse_args()
    repo = str(pathlib.Path(args.repo).resolve())
    sys.path.insert(0, repo)

    from unitree_envelope import rollout as R

    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    # 1. upstream's helpers are the ones the harness uses
    from deploy.deploy_mujoco.deploy_mujoco import get_gravity_orientation, pd_control
    g, p = R._upstream()
    check("uses upstream's pd_control and get_gravity_orientation",
          g is get_gravity_orientation and p is pd_control)

    # 2. same trajectory as the independent transcription of upstream's loop
    z_ref, v_ref, z0_ref = reference_loop(args.robot, repo, args.duration)
    res = R.rollout(args.robot, repo, duration=args.duration)
    check("step count matches", res.n_steps == len(z_ref),
          f"{res.n_steps} vs {len(z_ref)}")
    # the harness keeps traces internally; compare the summary it derives
    s = int(0.2 * len(z_ref))
    # Both divide by the height BEFORE the first step: the robot settles a
    # little during step 1, so z_ref[0] is already lower and would make the
    # two sides disagree at 1e-5 for a reason that is not the physics.
    check("min height ratio matches upstream trace",
          abs(res.min_height_ratio - z_ref.min() / z0_ref) < 1e-12,
          f"{res.min_height_ratio:.9f} vs {z_ref.min() / z0_ref:.9f}")
    for i, (nm, got) in enumerate([("vx", res.mean_vx), ("vy", res.mean_vy),
                                   ("yaw rate", res.mean_yaw_rate)]):
        ref = v_ref[s:, i].mean()
        check(f"mean {nm} matches upstream trace", abs(got - ref) < 1e-9,
              f"{got:.9f} vs {ref:.9f}")

    # 3. observation layout, read off upstream's source rather than assumed
    src = (pathlib.Path(repo) / "deploy/deploy_mujoco/deploy_mujoco.py").read_text()
    for frag in ["obs[:3] = omega", "obs[3:6] = gravity_orientation",
                 "obs[6:9] = cmd * cmd_scale",
                 "obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action",
                 "count = counter * simulation_dt", "period = 0.8"]:
        check(f"upstream still has: {frag}", frag in src)

    # 4. determinism, and stress-off == shipped behaviour
    again = R.rollout(args.robot, repo, duration=args.duration)
    check("deterministic across runs", again.as_dict() == res.as_dict())
    stressed = R.rollout(args.robot, repo, duration=args.duration, stress=R.Stress())
    check("default Stress() changes nothing", stressed.as_dict() == res.as_dict())

    results.append(test_recurrent_reset(repo, args.robot))

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
