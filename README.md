# What can Unitree's shipped walking policies actually do?

[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) ships
pre-trained walking policies for the G1, H1 and H1-2 humanoids, a MuJoCo
deployment script, and GIFs. It publishes **no numbers at all** — and the
script only ever sends one command, `cmd_init: [0.5, 0, 0]`, at wall-clock
speed inside a viewer.

This measures them: 1,050 headless rollouts across the command space and four
deployment stresses. Along the way it turned up something the upstream repo
does not document — **the shipped policies are recurrent**, and that has
practical consequences.

## Scope, up front

Training is not replicated. That needs Isaac Gym, which wants Linux with an
older CUDA than a current Blackwell GPU provides, and the available box is
shared and busy. Nothing here retrains anything; these are the shipped
checkpoints, evaluated as released. Everything runs on CPU — a 60-second
rollout takes about 0.5 s, which is what makes 1,050 of them practical.

## The harness is upstream's loop with the viewer removed

`deploy_mujoco.py` paces itself to the wall clock, so 60 s of simulation costs
60 s and the output is a GIF. `unitree_envelope/rollout.py` keeps the loop and
drops the viewer and the sleep. To make "keeps the loop" checkable rather than
asserted, `pd_control` and `get_gravity_orientation` are **imported** from
upstream rather than reimplemented, and `tests/test_matches_upstream.py`
holds a second implementation of the loop, transcribed from upstream's source
and sharing no code with the harness, which must agree:

```
15/15 passed      # including: mean vx / vy / yaw match upstream's trace to 1e-9
```

Two of my own bugs were caught this way, which is the reason the test exists:
storing traces in float32 disagreed with upstream at 1e-5 on min height, and
the observation's initial-height convention differed by one simulation step.

## Finding 1: the policies are LSTMs, and nothing says so

Each `motion.pt` carries `hidden_state` and `cell_state` buffers of shape
(1, 1, 64). **The action depends on history, not just on the 47-dim
observation.** Calling the policy twice on the same input returns different
answers.

Upstream's script loads the policy once and runs one episode, so it never has
to reset them, and the README does not mention them. Anything that runs more
than one episode in a process must reset them explicitly. When this harness
first cached a loaded policy across rollouts, the leak changed outcomes —
including whether a given seed fell — producing non-monotonic nonsense like
60% survival at an *easier* friction setting than 100% at a harder one. All
numbers below are from after the fix, and
`tests/test_matches_upstream.py::test_recurrent_reset` pins it.

If you are deploying these on hardware, this is the practical note: reset the
recurrent state when you restart the controller, or the robot begins each run
remembering the end of the last one.

## Finding 2: the single shipped command undersells them, except sideways

5 seeds per cell (seeds differ only in a 0.02 rad per-joint reset jitter),
20 s per rollout, 660 rollouts. Achieved velocity, survival in brackets.

**Forward — tracks well out to 3× the demoed command:**

| commanded vx | G1 | H1 | H1-2 |
|---|---|---|---|
| −0.50 | −0.438 (100%) | −0.548 (100%) | −0.366 (100%) |
| 0.00 | 0.011 (100%) | −0.050 (100%) | 0.002 (100%) |
| **0.50** (shipped) | **0.468** (100%) | **0.471** (100%) | **0.453** (100%) |
| 1.00 | 0.888 (100%) | 0.758 (100%) | 0.821 (100%) |
| 1.50 | 1.286 (100%) | 1.352 (100%) | 1.288 (100%) |

**Yaw — also fine, ±1.0 rad/s:** G1 reaches −0.905 / +0.862 for ∓1.0, H1
−0.905 / +0.913, H1-2 −0.813 / +0.898.

**Lateral — this is the weak axis, and H1-2 is biased:**

| commanded vy | G1 | H1 | H1-2 |
|---|---|---|---|
| −0.50 | −0.555 | −0.620 | **−0.126** |
| 0.00 | −0.081 | 0.010 | **+0.176** |
| +0.50 | 0.230 | 0.446 | 0.598 |

H1-2 commanded to hold zero lateral velocity **drifts sideways at 0.18 m/s**,
and asked for −0.5 delivers a quarter of it while +0.5 comes out slightly
over. That is a directional bias, not symmetric sloppiness. G1 drifts the
other way (−0.08 at zero) and reaches under half of +0.5.

Across the whole 660-rollout grid, **survival is 98.5%**, and every failure is
one case: **H1 at vx 1.5 combined with vy ≥ +0.25**, where all 5 seeds fall in
both cells. Nothing falls at negative vy, so that too is asymmetric.

So: staying upright is nearly free across this command space, and survival
alone would read as "works everywhere". Tracking is where the policies differ,
and lateral tracking is where they are weakest.

## Finding 3: observation latency is the binding constraint

Stress applied one axis at a time at the shipped command, 5 seeds, 375
rollouts. Survival:

| stress | G1 | H1 | H1-2 |
|---|---|---|---|
| **latency** (control steps @ 50 Hz) | ok to 2 (40 ms), **falls at 3** | ok to 1 (20 ms), **falls at 2** | ok to 2 (40 ms), **falls at 3** |
| observation noise (σ) | ok to 0.1, falls at 0.2 | 80% at 0.1, falls at 0.2 | ok to 0.1, falls at 0.2 |
| payload on the trunk | ok to 15 kg, falls at 20 | ok to 15, 80% at 20 | ok to 20 kg |
| friction scale | ok to 0.2×, falls at 0.1× | ok to 0.3×, falls at 0.2× | ok to 0.3×, falls at 0.2× |

The cliffs are sharp, and latency is the one that matters: **these policies
tolerate 20–40 ms of observation delay and then fall over.** Real sensing and
communication budgets sit in exactly that range, so it is the first thing to
measure on hardware. H1 is twice as sensitive as the other two.

The other three axes are comfortable by comparison — 15 kg of payload and a
floor five times slipperier than nominal are both survivable.

## Reproducing

```bash
bash scripts/setup.sh                                   # pinned checkout + the faithfulness test
python -m unitree_envelope.sweep baseline               # 15 rollouts,  ~3 s
python -m unitree_envelope.sweep commands               # 660 rollouts, ~64 s
python -m unitree_envelope.sweep stress                 # 375 rollouts, ~34 s
```

`results/*.json` hold every rollout, not just the summaries.

## What this does and does not establish

- **Simulation only, and the same simulator the policies are deployed in
  here.** A MuJoCo latency cliff at 40 ms is evidence about the policy, not a
  hardware measurement; real robots add actuator dynamics these models do not.
- **Survival is a coarse metric** — root height below 60% of its starting
  value. It says the robot stayed up, not that it walked well.
- **5 seeds per cell**, differing only in reset jitter. That is enough to
  separate 0% from 100%, which is how most of these cells land, but not to
  resolve an 80% from a 100%.
- **One stress axis at a time.** Latency plus payload together is untested and
  will not be the minimum of the two.
- Thresholds are the coarse grid listed above, so "falls at 3 steps" means
  between 2 and 3, not exactly 3.

## Attribution

Upstream is [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
(BSD 3-Clause, © Unitree Robotics), pinned at commit `276801e`. Robot models,
configs and the pre-trained policies are used from that checkout, which
`scripts/setup.sh` clones — nothing from it is redistributed here. Upstream's
`pd_control` and `get_gravity_orientation` are imported, not copied.
