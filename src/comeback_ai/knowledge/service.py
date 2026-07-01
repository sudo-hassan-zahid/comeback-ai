import httpx

from comeback_ai.domain.schemas import GuidanceRequest, GuidanceResponse, Source
from comeback_ai.knowledge.retriever import KnowledgeRetriever


class GuidanceService:
    def __init__(self, retriever: KnowledgeRetriever, api_key: str | None, model: str):
        self.retriever = retriever
        self.api_key = api_key
        self.model = model

    async def answer(self, request: GuidanceRequest) -> GuidanceResponse:
        query = f"{request.question} student risk {request.risk_level or ''}"
        matches = self.retriever.search(query)
        sources = [
            Source(title=chunk.title, section=chunk.section, score=round(score, 3))
            for chunk, score in matches
        ]
        if not matches:
            return GuidanceResponse(
                answer=(
                    "I could not find a grounded answer in the local support guide. "
                    "Please ask a teacher, adviser, or student-support office directly."
                ),
                sources=[],
                generated_by="local-retrieval",
            )
        if self.api_key:
            try:
                answer = await self._groq_answer(request.question, matches)
                return GuidanceResponse(answer=answer, sources=sources, generated_by="groq")
            except (httpx.HTTPError, KeyError, IndexError):
                # Guidance remains available when a free-tier key expires or hits a rate limit.
                pass
        bullets = "\n".join(f"- {chunk.text}" for chunk, _ in matches)
        return GuidanceResponse(
            answer=f"Here are the most relevant steps from the local support guide:\n{bullets}",
            sources=sources,
            generated_by="local-retrieval",
        )

    async def _groq_answer(self, question: str, matches: list) -> str:
        context = "\n\n".join(f"[{chunk.section}] {chunk.text}" for chunk, _ in matches)
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a student-support assistant. Answer only from the supplied "
                        "context. Be practical, kind, concise, and never make disciplinary, "
                        "medical, or admissions decisions. Say when context is insufficient."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        }
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
