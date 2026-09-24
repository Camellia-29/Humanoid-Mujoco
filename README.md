# humanoid-sim

A learning sandbox for **embodied AI / humanoid-robot locomotion** built on
[MuJoCo](https://mujoco.readthedocs.io/) + [Gymnasium](https://gymnasium.farama.org/).

The project is intentionally minimal: it starts from "look at the environment",
and grows step by step into a fully trained walking humanoid (PPO / SB3).

## Status

| Stage | Topic | Status |
|-------|-------|--------|
| 0 | Environment inspection / 3D visualization / model anatomy | ✅ |
| 1 | Engineering scaffold (README, requirements, .gitignore, dirs, git) | ✅ |
| 2 | Deep learning stack (PyTorch + Stable-Baselines3 + TensorBoard) | ✅ |
| 3 | Train a PPO policy on Humanoid-v5 | ⏳ planned |
| 4 | Evaluate, record video, plot learning curves | ⏳ planned |
| 5 | Custom XML / reward shaping / curriculum | ⏳ planned |
| 6 | Imitation learning (lerobot / robomimic) | ⏳ optional |

## Environment

- OS: Windows (cmd.exe)
- Python: 3.12.4 (Anaconda base, used as the venv host)
- Virtualenv: `.venv/` (Anaconda Python, `include-system-site-packages = false`)
- GPU: none → PyTorch CPU build

## Setup

```cmd
:: 1. Use the project venv (already created at .venv\)
D:\humanoid-sim\.venv\Scripts\python.exe --version

:: 2. Install remaining training dependencies
D:\humanoid-sim\.venv\Scripts\python.exe -m pip install -r requirements.txt

:: 3. Verify
D:\humanoid-sim\.venv\Scripts\python.exe -c "import torch, gymnasium, mujoco, stable_baselines3, tensorboard; print('all good')"
```

> If you later get an NVIDIA GPU, reinstall torch with CUDA:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`
> (Pick the CUDA version matching your driver.)

## Run the demos

```cmd
D:\humanoid-sim\.venv\Scripts\python.exe scripts\01_environment.py
D:\humanoid-sim\.venv\Scripts\python.exe scripts\02_visualization.py
D:\humanoid-sim\.venv\Scripts\python.exe scripts\03_model.py
```

### Demo map

| Script | What you learn |
|---------|---------------|
| `01_environment.py` | How `gym.make("Humanoid-v5")` works; obs (348) / action (17) shape; what a random-policy rollout looks like |
| `02_visualization.py` | Open an interactive 3D MuJoCo viewer; mouse-drag / scroll / space-pause |
| `03_model.py` | Walk through every body, joint, actuator in `models/humanoid.xml` |

## Layout

```
humanoid-sim/
├── .venv/                  # virtual environment (gitignored)
├── .gitignore
├── README.md
├── requirements.txt
├── models/                 # MuJoCo XML model files
│   └── humanoid.xml
├── scripts/                # numbered, self-contained learning scripts
│   ├── 01_environment.py
│   ├── 02_visualization.py
│   └── 03_model.py
├── envs/                   # custom Gymnasium env wrappers / reward shaping
├── agents/                 # policy / value networks (later)
├── common/                 # shared utils (later)
├── configs/                # experiment configs (later)
├── logs/                   # tensorboard / text logs (gitignored)
├── checkpoints/            # saved model weights (gitignored)
├── videos/                 # recorded rollouts (gitignored)
└── tests/                  # smoke / unit tests (later)
```

## Roadmap

- `04_train_ppo.py` — wrap `Humanoid-v5` in `VectorEnv`, train PPO from SB3, save checkpoints + TensorBoard logs.
- `05_eval.py` — load a checkpoint, run `RecordVideo`, plot reward curve with matplotlib.
- `06_custom_env.py` — wrap the env to shape reward (alive-bonus, forward-velocity bonus, energy penalty).
- `07_curriculum.py` — gradually increase difficulty.
- Imitation learning entry (lerobot) once enough demo data is collected.

## Conventions

- All scripts are runnable as `python scripts/NN_xxx.py` from the project root.
- Models referenced via **relative paths** built from `__file__` — never hard-code `D:/...` again.
- Every script is self-contained and prints what it does; pedagogical clarity over abstraction.
- Dependencies are pinned in `requirements.txt`; the venv is the single source of truth.

## References

- [MuJoCo docs](https://mujoco.readthedocs.io/)
- [Gymnasium docs](https://gymnasium.farama.org/)
- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/)
- [Humanoid-v5 source](https://gymnasium.farama.org/environments/mujoco/humanoid/)