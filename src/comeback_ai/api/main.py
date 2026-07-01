from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from comeback_ai.config import get_settings
from comeback_ai.domain.schemas import (
    GuidanceRequest,
    GuidanceResponse,
    RiskResponse,
    StudentProfile,
)
from comeback_ai.knowledge.retriever import KnowledgeRetriever
from comeback_ai.knowledge.service import GuidanceService
from comeback_ai.ml.service import RiskService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.risk_service = RiskService(settings.artifact_dir)
    app.state.guidance_service = GuidanceService(
        KnowledgeRetriever(settings.knowledge_dir), settings.groq_api_key, settings.groq_model
    )
    yield


app = FastAPI(
    title="Comeback AI",
    description="Explainable academic-risk signals and grounded student support.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/v1/risk", response_model=RiskResponse)
def assess_risk(profile: StudentProfile, request: Request) -> RiskResponse:
    return request.app.state.risk_service.predict(profile)


@app.post("/v1/guidance", response_model=GuidanceResponse)
async def get_guidance(body: GuidanceRequest, request: Request) -> GuidanceResponse:
    return await request.app.state.guidance_service.answer(body)
