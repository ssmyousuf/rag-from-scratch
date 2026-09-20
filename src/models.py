from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    id: str
    title: str
    text: str
    metadata: dict[str, Any]


@dataclass
class Query:
    id: str
    text: str
    answer: str
    supporting_documents: list[str]