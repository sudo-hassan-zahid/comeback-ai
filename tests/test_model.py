from comeback_ai.domain.schemas import StudentProfile
from comeback_ai.ml.service import RiskService
from comeback_ai.ml.train import train


def test_training_and_prediction(tmp_path):
    card = train(tmp_path)
    service = RiskService(tmp_path)
    result = service.predict(
        StudentProfile(
            attendance_rate=55,
            assignment_completion=45,
            average_grade=48,
            study_hours_weekly=2,
            previous_failures=2,
            commute_minutes=70,
            has_internet=False,
            works_part_time=True,
            reports_stress=True,
            asked_for_help=False,
        )
    )
    assert card["selected_model"] in card["metrics"]
    assert 0 <= result.risk_probability <= 1
    assert len(result.top_factors) == 4
