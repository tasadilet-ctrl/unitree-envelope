#!/usr/bin/env bash
# Clones the pinned unitree_rl_gym checkout this repo measures. ~195 MB.
# Isaac Gym is not required: nothing here trains.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=276801e            # the commit every number in the README was produced against
mkdir -p third_party
[ -d third_party/unitree_rl_gym ] || \
  git clone -q https://github.com/unitreerobotics/unitree_rl_gym.git third_party/unitree_rl_gym
git -C third_party/unitree_rl_gym fetch -q --depth 1 origin "$COMMIT" 2>/dev/null \
  || git -C third_party/unitree_rl_gym fetch -q origin
git -C third_party/unitree_rl_gym checkout -q "$COMMIT"
echo "unitree_rl_gym at $(git -C third_party/unitree_rl_gym rev-parse --short HEAD)"
python3 tests/test_matches_upstream.py --repo third_party/unitree_rl_gym
