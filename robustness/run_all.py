"""Run every robustness experiment: .venv/bin/python -m robustness.run_all"""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ["exp_seeds", "exp_temporal", "exp_noise_missing", "exp_subgroups", "exp_threshold"]:
    print(f"=== {script} ===", flush=True)
    runpy.run_path(str(HERE / f"{script}.py"), run_name="__main__")
