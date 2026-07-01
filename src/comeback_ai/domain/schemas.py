from typing import Literal

from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    attendance_rate: float = Field(ge=0, le=100, examples=[72])
    assignment_completion: float = Field(ge=0, le=100, examples=[65])
    average_grade: float = Field(ge=0, le=100, examples=[58])
    study_hours_weekly: float = Field(ge=0, le=80, examples=[7])
    previous_failures: int = Field(ge=0, le=20, examples=[1])
    commute_minutes: int = Field(ge=0, le=300, examples=[45])
    has_internet: bool = True
    works_part_time: bool = False
    reports_stress: bool = False
    asked_for_help: bool = False


class Factor(BaseModel):
    feature: str
    label: str
    direction: Literal["increases risk", "reduces risk"]
    impact: float


class RiskResponse(BaseModel):
    risk_probability: float
    risk_level: Literal["low", "moderate", "high"]
    top_factors: list[Factor]
    note: str


class GuidanceRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    risk_level: Literal["low", "moderate", "high"] | None = None


class Source(BaseModel):
    title: str
    section: str
    score: float


class GuidanceResponse(BaseModel):
    answer: str
    sources: list[Source]
    generated_by: Literal["local-retrieval", "groq"]
