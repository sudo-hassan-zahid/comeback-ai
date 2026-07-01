from pathlib import Path

import joblib
import pandas as pd

from comeback_ai.domain.schemas import Factor, RiskResponse, StudentProfile
from comeback_ai.ml.data import FEATURES
from comeback_ai.ml.train import train

LABELS = {
    "attendance_rate": "Attendance rate",
    "assignment_completion": "Assignment completion",
    "average_grade": "Average grade",
    "study_hours_weekly": "Weekly study time",
    "previous_failures": "Previous failed courses",
    "commute_minutes": "Commute time",
    "has_internet": "Reliable internet access",
    "works_part_time": "Part-time work",
    "reports_stress": "Reported stress",
    "asked_for_help": "Asked for help",
}


class RiskService:
    def __init__(self, artifact_dir: Path):
        model_path = artifact_dir / "risk_model.joblib"
        if not model_path.exists():
            train(artifact_dir)
        self.model = joblib.load(model_path)

    def predict(self, profile: StudentProfile) -> RiskResponse:
        values = profile.model_dump()
        frame = pd.DataFrame([values], columns=FEATURES)
        probability = float(self.model.predict_proba(frame)[0, 1])
        level = "high" if probability >= 0.7 else "moderate" if probability >= 0.4 else "low"
        factors = self._explain(frame)
        return RiskResponse(
            risk_probability=round(probability, 4),
            risk_level=level,
            top_factors=factors,
            note="Support signal only—not a diagnosis or an automated decision.",
        )

    def _explain(self, frame: pd.DataFrame) -> list[Factor]:
        classifier = self.model.named_steps["classifier"]
        transformed = self.model.named_steps["preprocessor"].transform(frame)[0]
        if hasattr(classifier, "coef_"):
            impacts = transformed * classifier.coef_[0]
        else:
            # Global feature importance is a transparent fallback for the tree candidate.
            centered = transformed
            impacts = centered * classifier.feature_importances_
        ranked = sorted(
            zip(FEATURES, impacts, strict=True), key=lambda item: abs(item[1]), reverse=True
        )
        return [
            Factor(
                feature=name,
                label=LABELS[name],
                direction="increases risk" if impact > 0 else "reduces risk",
                impact=round(abs(float(impact)), 3),
            )
            for name, impact in ranked[:4]
        ]
