from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel

MARKDOWN_CHUNK_SIZE = 800
MARKDOWN_CHUNK_OVERLAP = 120


class ProcessText(BaseModel):
    document: str

    def markdown_chunking(self) -> list[str]:
        splitter = MarkdownTextSplitter(
            chunk_size=MARKDOWN_CHUNK_SIZE,
            chunk_overlap=MARKDOWN_CHUNK_OVERLAP,
        )
        return splitter.split_text(self.document)

    def recursive_chunking(self) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=256, chunk_overlap=48, separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(self.document)

    def md_splitter(self) -> list[Document]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(self.document)
        return md_header_splits


if __name__ == "__main__":
    markdown_file_path = Path("../../data/parsed/synthetic_demo.md")
    document_content = markdown_file_path.read_text(encoding="utf-8")

    processor = ProcessText(document=document_content)
    # chunks = processor.recursive_chunking()
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"--- Chunk {i} ---")
    #     print(chunk)
    #     print()

    chunks = processor.md_splitter()
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
        print()
