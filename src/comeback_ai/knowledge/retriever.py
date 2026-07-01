import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    title: str
    section: str
    text: str


class KnowledgeRetriever:
    def __init__(self, directory: Path):
        self.chunks = self._load(directory)
        if not self.chunks:
            raise ValueError(f"No Markdown knowledge documents found in {directory}")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(
            [f"{chunk.title} {chunk.section} {chunk.text}" for chunk in self.chunks]
        )

    @staticmethod
    def _load(directory: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in sorted(directory.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            title = path.stem.replace("-", " ").title()
            sections = re.split(r"\n(?=## )", content)
            for section in sections:
                lines = section.strip().splitlines()
                heading = lines[0].lstrip("# ") if lines else title
                body = "\n".join(lines[1:]).strip()
                if body:
                    chunks.append(Chunk(title=title, section=heading, text=body))
        return chunks

    def search(self, query: str, limit: int = 3) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        indices = scores.argsort()[::-1][:limit]
        return [
            (self.chunks[index], float(scores[index])) for index in indices if scores[index] > 0
        ]
