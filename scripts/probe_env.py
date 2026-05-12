# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""Probe Python environment for required packages."""
import importlib
import importlib.util

MODS = [
    "torch", "mne", "mne_icalabel", "moabb", "numpy", "scipy", "sklearn",
    "gymnasium", "d3rlpy", "transformers", "peft", "matplotlib", "seaborn",
    "yaml", "tqdm", "rich",
]

for m in MODS:
    spec = importlib.util.find_spec(m)
    if spec is None:
        print(f"{m:20s} MISSING")
        continue
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "OK")
        print(f"{m:20s} {ver}")
    except Exception as e:
        print(f"{m:20s} IMPORT-ERROR: {type(e).__name__}: {e}")
