from fastapi.testclient import TestClient

from comeback_ai.api.main import app


def test_health_and_risk(monkeypatch, tmp_path):
    monkeypatch.setenv("COMEBACK_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("COMEBACK_KNOWLEDGE_DIR", "knowledge")
    from comeback_ai.config import get_settings

    get_settings.cache_clear()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "healthy"}
        response = client.post(
            "/v1/risk",
            json={
                "attendance_rate": 80,
                "assignment_completion": 85,
                "average_grade": 74,
                "study_hours_weekly": 8,
                "previous_failures": 0,
                "commute_minutes": 20,
                "has_internet": True,
                "works_part_time": False,
                "reports_stress": False,
                "asked_for_help": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["risk_level"] in {"low", "moderate", "high"}

        guidance = client.post(
            "/v1/guidance",
            json={"question": "What can I do when I have unreliable internet?"},
        )
        assert guidance.status_code == 200
        assert guidance.json()["generated_by"] == "local-retrieval"
        assert guidance.json()["sources"][0]["section"] == "Internet access"
