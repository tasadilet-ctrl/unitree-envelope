"""Headless rollout of Unitree's shipped MuJoCo policies, with stress knobs.

Upstream's deploy/deploy_mujoco/deploy_mujoco.py is a viewer script paced to
the wall clock: 60 s of simulation takes 60 s of real time and the results are
a GIF. The control loop here is the same loop, with the viewer and the sleep
removed, so 60 s of simulation takes about half a second and thousands of
rollouts are affordable.

Faithfulness: pd_control and get_gravity_orientation are IMPORTED from
upstream's module rather than reimplemented, and the loop keeps upstream's
order exactly -- PD torque at every sim step, policy every control_decimation
steps, the gait phase from counter * simulation_dt, and the previous action in
the observation. tests/test_matches_upstream.py pins the observation layout and
the phase convention.

Every stress knob defaults to off, so a default rollout is upstream's
behaviour.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Optional, Sequence

import mujoco
import numpy as np
import torch
import yaml

ROBOTS = ("g1", "h1", "h1_2")


def _upstream(tapnet_root=None):
    """Import upstream's two control helpers, so they are not reimplemented."""
    from deploy.deploy_mujoco.deploy_mujoco import (  # noqa
        get_gravity_orientation, pd_control)
    return get_gravity_orientation, pd_control


@dataclasses.dataclass
class Stress:
    """Deployment-realistic perturbations. All zero/one means 'as shipped'."""
    obs_latency_steps: int = 0        # control steps of delay on the observation
    obs_noise_std: float = 0.0        # gaussian noise, in observation units
    added_trunk_mass: float = 0.0     # kg on the root body
    friction_scale: float = 1.0       # multiplies every geom's sliding friction
    init_qpos_noise: float = 0.0      # per-joint rad jitter at reset
    init_yaw: float = 0.0             # initial heading, rad
    seed: int = 0                     # explicit: see the note in rollout()


@dataclasses.dataclass
class Result:
    robot: str
    cmd: tuple
    survived: bool
    fall_time: Optional[float]
    duration: float
    mean_vx: float
    mean_vy: float
    mean_yaw_rate: float
    track_err_vx: float
    track_err_vy: float
    track_err_yaw: float
    min_height_ratio: float
    mean_height: float
    n_steps: int

    def as_dict(self):
        return dataclasses.asdict(self)


def reset_policy_state(policy, init_state: dict) -> None:
    """Restore the policy's recurrent buffers to their load-time values.

    Required between episodes: these policies are LSTMs whose hidden_state and
    cell_state persist across calls.
    """
    with torch.no_grad():
        for name, buf in policy.named_buffers():
            if name in init_state:
                buf.copy_(init_state[name])


def load_config(robot: str, repo_root) -> dict:
    if robot not in ROBOTS:
        raise ValueError(f"robot must be one of {ROBOTS}, got {robot!r}")
    root = str(repo_root)
    p = pathlib.Path(root) / "deploy" / "deploy_mujoco" / "configs" / f"{robot}.yaml"
    c = yaml.safe_load(p.read_text())
    for k in ("policy_path", "xml_path"):
        c[k] = c[k].replace("{LEGGED_GYM_ROOT_DIR}", root)
    return c


def rollout(robot: str, repo_root, cmd: Sequence[float] = None,
            stress: Stress = None, duration: float = None,
            fall_ratio: float = 0.6, settle_frac: float = 0.2,
            policy_cache: dict = None) -> Result:
    """Run one episode. Returns survival and command-tracking metrics.

    fall_ratio: a fall is the root dropping below this fraction of its initial
    height -- relative, so one threshold works for G1 (0.79 m) and H1 (1.03 m).
    settle_frac: the leading fraction of the episode excluded from the tracking
    averages, so the start-up transient is not scored as tracking error.
    """
    get_gravity_orientation, pd_control = _upstream()
    stress = stress or Stress()
    c = load_config(robot, repo_root)
    cmd = np.array(c["cmd_init"] if cmd is None else cmd, dtype=np.float32)

    m = mujoco.MjModel.from_xml_path(c["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = c["simulation_dt"]

    if stress.added_trunk_mass:
        m.body_mass[1] += stress.added_trunk_mass      # body 0 is the world
    if stress.friction_scale != 1.0:
        m.geom_friction[:, 0] *= stress.friction_scale

    # Explicit integer seed, NOT derived from hash() of anything containing a
    # string: Python salts string hashing per process, so a hash-derived seed
    # gives different jitter in a worker than in the parent, and sweeps would
    # not reproduce. tests/ checks this across processes.
    rng = np.random.default_rng(stress.seed)
    if stress.init_qpos_noise:
        d.qpos[7:] += rng.normal(0, stress.init_qpos_noise, size=m.nq - 7)
    if stress.init_yaw:
        half = stress.init_yaw / 2.0
        d.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]

    # The shipped policies are RECURRENT: each carries hidden_state and
    # cell_state buffers (1,1,64), so the action depends on history, not just
    # on the current observation. Upstream's viewer script loads the policy
    # once and runs a single episode, so it never has to reset them. Reusing a
    # loaded module across episodes leaks the previous episode's memory into
    # the next -- which silently changed outcomes here, including whether a
    # seed fell. The load-time buffers are snapshotted and restored per
    # episode, which keeps the cache's speed without the leak.
    key = (robot, c["policy_path"])
    if policy_cache is not None and key in policy_cache:
        policy, init_state = policy_cache[key]
    else:
        policy = torch.jit.load(c["policy_path"])
        policy.eval()
        init_state = {n: b.detach().clone() for n, b in policy.named_buffers()}
        if policy_cache is not None:
            policy_cache[key] = (policy, init_state)
    reset_policy_state(policy, init_state)

    kps = np.array(c["kps"], np.float32)
    kds = np.array(c["kds"], np.float32)
    default_angles = np.array(c["default_angles"], np.float32)
    cmd_scale = np.array(c["cmd_scale"], np.float32)
    na, no = c["num_actions"], c["num_obs"]
    dt, dec = c["simulation_dt"], c["control_decimation"]
    duration = c["simulation_duration"] if duration is None else duration

    action = np.zeros(na, np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(no, np.float32)
    obs_buffer: list = []

    z0 = float(d.qpos[2])
    n_steps = int(duration / dt)
    # float64: these feed the reported metrics, and float32 storage alone was
    # enough to disagree with upstream's trace at 1e-5 on min height.
    zs = np.empty(n_steps, np.float64)
    vel = np.empty((n_steps, 3), np.float64)
    fall_time = None

    for i in range(n_steps):
        tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                         np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        counter = i + 1

        if counter % dec == 0:
            qj = (d.qpos[7:] - default_angles) * c["dof_pos_scale"]
            dqj = d.qvel[6:] * c["dof_vel_scale"]
            omega = d.qvel[3:6] * c["ang_vel_scale"]
            phase = (counter * dt) % 0.8 / 0.8
            obs[:3] = omega
            obs[3:6] = get_gravity_orientation(d.qpos[3:7])
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + na] = qj
            obs[9 + na:9 + 2 * na] = dqj
            obs[9 + 2 * na:9 + 3 * na] = action
            obs[9 + 3 * na:9 + 3 * na + 2] = [np.sin(2 * np.pi * phase),
                                              np.cos(2 * np.pi * phase)]
            fed = obs
            if stress.obs_noise_std:
                fed = fed + rng.normal(0, stress.obs_noise_std, size=no).astype(np.float32)
            if stress.obs_latency_steps:
                obs_buffer.append(fed.copy())
                # Before the buffer fills, the oldest available observation is
                # used -- the same thing a real robot does at start-up.
                fed = obs_buffer[max(0, len(obs_buffer) - 1 - stress.obs_latency_steps)]
            with torch.no_grad():
                action = policy(torch.from_numpy(np.ascontiguousarray(fed)).unsqueeze(0)
                                ).numpy().squeeze()
            target_dof_pos = action * c["action_scale"] + default_angles

        zs[i] = d.qpos[2]
        # (vx, vy, yaw rate) -- qvel[5] is yaw, not vz
        vel[i] = (d.qvel[0], d.qvel[1], d.qvel[5])
        if fall_time is None and d.qpos[2] < fall_ratio * z0:
            fall_time = counter * dt
            zs = zs[:i + 1]
            vel = vel[:i + 1]
            break

    s = int(settle_frac * len(zs))
    window = vel[s:] if len(vel) > s else vel
    mean = window.mean(axis=0) if len(window) else np.zeros(3)
    return Result(
        robot=robot, cmd=tuple(float(x) for x in cmd),
        survived=fall_time is None, fall_time=fall_time,
        duration=float(len(zs) * dt),
        mean_vx=float(mean[0]), mean_vy=float(mean[1]), mean_yaw_rate=float(mean[2]),
        track_err_vx=float(mean[0] - cmd[0]),
        track_err_vy=float(mean[1] - cmd[1]),
        track_err_yaw=float(mean[2] - cmd[2]),
        min_height_ratio=float(zs.min() / z0), mean_height=float(zs.mean()),
        n_steps=len(zs))
