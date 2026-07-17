"""
scripts/run_cohort.py
----------------------
End-to-end cohort extraction: MIMIC-IV -> labeled cohort -> train/val/test splits.

Run from project root:
    python scripts/run_cohort.py

On Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python scripts/run_cohort.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cohort.extract_cohort import build_cohort
from src.cohort.split import split_cohort
from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg.cohort.random_seed)

    log.info("Starting cohort extraction ...")
    cohort = build_cohort(cfg)

    log.info("\nCohort summary:")
    log.info("  Total admissions : %d", len(cohort))
    log.info("  Unique patients  : %d", cohort["subject_id"].nunique())
    log.info("  Readmit rate     : %.1f%%", 100 * cohort["readmitted_30d"].mean())
    log.info("  Competing events : %.1f%%", 100 * cohort["competing_event"].mean())

    log.info("\nSplitting cohort ...")
    train_df, val_df, test_df = split_cohort(cohort, cfg)

    log.info("\nCohort extraction complete [OK]")
    log.info("Files saved to: %s", cfg.paths.cohort_dir)


if __name__ == "__main__":
    main()
