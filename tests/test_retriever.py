from pathlib import Path

from comeback_ai.knowledge.retriever import KnowledgeRetriever


def test_retriever_finds_internet_support():
    retriever = KnowledgeRetriever(Path("knowledge"))
    matches = retriever.search("I have no reliable internet connection")
    assert matches
    assert matches[0][0].section == "Internet access"
