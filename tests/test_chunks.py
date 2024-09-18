# mypy: allow-untyped-defs

from simulst.text_chunks import ConcatenatedText, TextChunk


def test_concatenated_chunks_2():
    chunks = [TextChunk("Границы моего языка."), TextChunk("Языка определяют границы моего сознания.")]

    assert ConcatenatedText(chunks) == TextChunk("Границы моего языка определяют границы моего сознания.")


def test_concatenated_chunks_3():
    chunks = [
        TextChunk("Границы моего языка !!!"),
        TextChunk("Языка определяют границы моего сознания..."),
        TextChunk("Сознания определяют границы моего языка."),
    ]

    assert ConcatenatedText(chunks) == TextChunk(
        "Границы моего языка определяют границы моего сознания определяют границы моего языка."
    )


def test_concatenated_chunks_4():
    chunks = [
        TextChunk("Границы моего языка"),
        TextChunk("определяют границы моего сознания."),
    ]

    assert ConcatenatedText(chunks) == TextChunk("Границы моего языка определяют границы моего сознания.")
