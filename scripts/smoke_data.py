# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""Smoke test: preprocess one subject and report shapes / stats."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import preprocess_subject  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("smoke_data")


def main():
    # BCI-IV-2b is the smallest dataset (3 channels, ~50 MB total) — fastest smoke.
    dataset_key = "bci2b"
    subject_id = 1
    log.info("Preprocessing %s subject %d ...", dataset_key, subject_id)
    tb = preprocess_subject(dataset_key, subject_id)
    log.info(
        "Done. n_trials=%d  n_channels=%d  n_times=%d  sfreq=%.1f",
        tb.n_trials, tb.n_channels, tb.n_times, tb.sfreq,
    )
    log.info("Class labels: %s  Class counts: %s",
             tb.class_labels,
             {c: int((tb.y == i).sum()) for i, c in enumerate(tb.class_labels)})
    log.info("Sessions present: %s", sorted(set(tb.session.tolist())))
    log.info("X dtype=%s  X.mean=%.4f  X.std=%.4f",
             tb.X.dtype, float(tb.X.mean()), float(tb.X.std()))
    log.info("Channel names: %s", tb.ch_names)

    # Sanity asserts
    assert tb.X.shape == (tb.n_trials, tb.n_channels, tb.n_times)
    assert tb.X.dtype.name == "float32"
    assert tb.y.dtype.name == "int64"
    log.info("Smoke OK.")


if __name__ == "__main__":
    main()
