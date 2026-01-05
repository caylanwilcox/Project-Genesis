"""
SPY Daily Range Plan Prediction System

ML-driven intraday trading system that generates calibrated
probability estimates for SPY price targets.
"""

__version__ = "1.0.0"

from .config import (
    DataConfig,
    ModelConfig,
    TargetConfig,
    WalkForwardConfig,
    TARGET_LABELS,
    FIRST_TOUCH_CLASSES,
    ALL_FEATURES,
)

from .inference import (
    PlanGenerator,
    DailyRangePlan,
    infer_today,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TargetConfig",
    "WalkForwardConfig",
    "TARGET_LABELS",
    "FIRST_TOUCH_CLASSES",
    "ALL_FEATURES",
    "PlanGenerator",
    "DailyRangePlan",
    "infer_today",
]
