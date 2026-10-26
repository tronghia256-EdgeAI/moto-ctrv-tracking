"""
tests/synthetic.py
──────────────────
Thin re-export: the trajectory generators live in code/utils/synthetic.py
so the experiment harness (code/scripts/run_experiment.py --synthetic)
shares one source of truth with the tests.
"""

from code.utils.synthetic import (  # noqa: F401
    CLASS_CAR,
    CLASS_MOTORCYCLE,
    CLASS_BUS,
    CLASS_TRUCK,
    straight,
    constant_turn,
    weaving,
    centers_to_bboxes,
    to_detections,
)
