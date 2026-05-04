from rag.chunking.process_text import MARKDOWN_CHUNK_SIZE, ProcessText


def test_markdown_chunking_keeps_short_table_together():
    text = (
        "## Wyniki laboratoryjne\n\n"
        "|Badanie|Wynik|Jedn.|MIN|MAX|\n"
        "|---|---:|---|---:|---:|\n"
        "|ALT (ICD-9: 117)|15|U/l|0|41|\n"
        "|CRP (ICD-9: 181)|0,3|mg/l|0,0|5,0|\n\n"
        "Wniosek: wyniki w zakresie referencyjnym."
    )
    processor = ProcessText(document=text)

    chunks = processor.markdown_chunking()

    assert len(chunks) == 1
    assert "ALT (ICD-9: 117)" in chunks[0]
    assert "CRP (ICD-9: 181)" in chunks[0]
    assert "Wniosek" in chunks[0]


def test_markdown_chunking_splits_long_text_without_tiny_chunks():
    text = " ".join(
        f"Sentence {index} describes clinical finding and treatment response."
        for index in range(80)
    )
    processor = ProcessText(document=text)

    chunks = processor.markdown_chunking()

    assert len(chunks) > 1
    assert max(len(chunk) for chunk in chunks) <= MARKDOWN_CHUNK_SIZE
    assert min(len(chunk) for chunk in chunks) >= 80
