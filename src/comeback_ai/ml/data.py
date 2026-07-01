import numpy as np
import pandas as pd

FEATURES = [
    "attendance_rate",
    "assignment_completion",
    "average_grade",
    "study_hours_weekly",
    "previous_failures",
    "commute_minutes",
    "has_internet",
    "works_part_time",
    "reports_stress",
    "asked_for_help",
]


def generate_demo_data(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Create reproducible, explicitly synthetic data for learning and demos."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "attendance_rate": np.clip(rng.normal(78, 15, rows), 20, 100),
            "assignment_completion": np.clip(rng.normal(75, 20, rows), 0, 100),
            "average_grade": np.clip(rng.normal(67, 16, rows), 0, 100),
            "study_hours_weekly": np.clip(rng.gamma(2.2, 3, rows), 0, 40),
            "previous_failures": np.clip(rng.poisson(0.6, rows), 0, 5),
            "commute_minutes": np.clip(rng.gamma(2, 18, rows), 0, 150),
            "has_internet": rng.binomial(1, 0.82, rows),
            "works_part_time": rng.binomial(1, 0.28, rows),
            "reports_stress": rng.binomial(1, 0.38, rows),
            "asked_for_help": rng.binomial(1, 0.42, rows),
        }
    )
    log_odds = (
        -0.055 * (frame.attendance_rate - 70)
        - 0.035 * (frame.assignment_completion - 65)
        - 0.025 * (frame.average_grade - 55)
        - 0.07 * frame.study_hours_weekly
        + 0.75 * frame.previous_failures
        + 0.008 * frame.commute_minutes
        - 0.45 * frame.has_internet
        + 0.35 * frame.works_part_time
        + 0.65 * frame.reports_stress
        - 0.4 * frame.asked_for_help
        - 0.35
    )
    probability = 1 / (1 + np.exp(-log_odds))
    frame["at_risk"] = rng.binomial(1, probability)
    return frame
