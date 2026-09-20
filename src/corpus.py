from .models import Document


class Corpus:
    def __init__(self, documents: list[Document]):
        self.documents_by_id: dict[str, Document] = {}

        for document in documents:
            if document.id not in self.documents_by_id:
                self.documents_by_id[document.id] = document

        self.documents = list(self.documents_by_id.values())

    def get_document(self, document_id: str) -> Document | None:
        return self.documents_by_id.get(document_id)

    def __len__(self) -> int:
        return len(self.documents)