# mypy: allow-untyped-defs

from simulst.text_chunks import ConcatenatedText, TextChunk


def test_concatenated_chunks_2():
    chunks = [TextChunk("Границы моего языка."), TextChunk("Языка определяют границы моего сознания.")]

    assert ConcatenatedText(chunks) == TextChunk("Границы моего языка определяют границы моего сознания.")


def test_concatenated_chunks_3():
    # TODO: fix this test
    chunks = [
        TextChunk("Границы моего языка!!!"),
        TextChunk("Языка определяют границы моего сознания..."),
        TextChunk("   Сознания определяют границы моего языка."),
    ]

    assert ConcatenatedText(chunks) == TextChunk(
        "Границы моего языка определяют границы моего сознания определяют границы моего языка."
    )
