
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable
from langchain_core.documents import Document


class LoaderError(RuntimeError):
    """Raised when a source cannot be loaded or its format is unsupported."""


def clean_source_name(source: Any) -> str: # to only return the file name not my whole laptop
    if not source:
        return "unknown"
    normalized = str(source).replace("\\", "/")
    return Path(normalized).name or str(source)


class DocumentLoader:
    _TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown", ".rst", ".log"}
    _HTML_EXTENSIONS = {".html", ".htm"}
    _STRUCTURED_EXTENSIONS = {".json", ".csv"}
    _SUPPORTED_EXTENSIONS = (
        _TEXT_EXTENSIONS | _HTML_EXTENSIONS | _STRUCTURED_EXTENSIONS | {".pdf"}
    )

    def __init__(
        self,
        *,
        encoding: str = "utf-8",
    ) -> None:
        self.encoding = encoding

    def load(self, source: str | Path) -> list[Document]:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Document does not exist: {path}")
        return self._load_file(path)

    def load_many(self, sources: Iterable[str | Path]) -> list[Document]:
        
        documents: list[Document] = []
        for source in sources:
            documents.extend(self.load(source))
        return documents

    def load_directory( # to load all the files in my data directory
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
    ) -> list[Document]:
        
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {root}")

        paths = root.rglob("*") if recursive else root.glob("*")
        supported_paths = sorted(
            path for path in paths if path.is_file() and self._is_supported(path)
        )
        return self.load_many(supported_paths)

    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix in self._HTML_EXTENSIONS:
            return [self._document(self._html_to_text(path.read_text(encoding=self.encoding)), path)]
        if suffix in self._TEXT_EXTENSIONS:
            return [self._document(path.read_text(encoding=self.encoding), path)]
        if suffix == ".json":
            return [self._document(self._json_to_text(path), path)]
        if suffix == ".csv":
            return [self._document(self._csv_to_text(path), path)]
        raise LoaderError(
            f"Unsupported file type '{suffix or '[no extension]'}' for {path}. "
            f"Supported types: {', '.join(sorted(self._SUPPORTED_EXTENSIONS))}"
        )

    def _load_pdf(self, path: Path) -> list[Document]:
        try:
            import pdfplumber
        except ImportError as exc:
            raise LoaderError(
                "PDF loading requires the 'pdfplumber' package. "
                "Install it with: pip install pdfplumber"
            ) from exc

        documents: list[Document] = []
        try:
            with pdfplumber.open(path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(
                            self._document(
                                text,
                                path,
                                page=page_number,
                                total_pages=len(pdf.pages),
                            )
                        )
        except Exception as exc:
            raise LoaderError(f"Could not read PDF '{path}': {exc}") from exc
        return documents

    def _document(self, text: str, path: Path, **metadata: Any) -> Document:
        raw_source = metadata.pop("source", path.name)
        return Document(
            page_content=text.strip(),
            metadata={"source": clean_source_name(raw_source), "type": path.suffix.lower()[1:], **metadata},
        )

    @staticmethod
    def _html_to_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise LoaderError(
                "HTML loading requires the 'beautifulsoup4' package. "
                "Install it with: pip install beautifulsoup4"
            ) from exc

        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "noscript", "template"]):
            element.decompose()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))

    def _json_to_text(self, path: Path) -> str:
        try:
            value = json.loads(path.read_text(encoding=self.encoding))
        except json.JSONDecodeError as exc:
            raise LoaderError(f"Invalid JSON in '{path}': {exc}") from exc
        return json.dumps(value, ensure_ascii=False, indent=2)

    def _csv_to_text(self, path: Path) -> str:
        with path.open("r", encoding=self.encoding, newline="") as file:
            rows = list(csv.reader(file))
        return "\n".join(" | ".join(row) for row in rows)

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self._SUPPORTED_EXTENSIONS