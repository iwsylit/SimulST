from functools import reduce
from typing import Iterator, Self, Sequence

from simulst.translation import Translation


class TextChunk:
    def __init__(self, text: str) -> None:
        self._text = text
        self._clean_text = self._clean()

    @classmethod
    def from_translation(cls, translation: Translation) -> Self:
        return cls(translation.target)

    def _clean(self) -> str:
        return "".join(char for char in self.text if char.isalnum() or char.isspace()).lower()

    @property
    def text(self) -> str:
        return self._text

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, index: int | slice) -> str:
        return self.text[index]

    def __repr__(self) -> str:
        return f"TextChunk: {self.text}"

    def __eq__(self, other: Self) -> bool:  # type: ignore
        return self.text == other.text


class TextChunks:
    def __init__(self, chunks: Sequence[TextChunk]) -> None:
        self._chunks = list(chunks)

    def append(self, chunk: TextChunk) -> None:
        self._chunks.append(chunk)

    @property
    def chunks(self) -> list[TextChunk]:
        return self._chunks

    def __iter__(self) -> Iterator[TextChunk]:
        return iter(self._chunks)


class ConcatenatedText(TextChunks):
    _MAX_OVERLAP = 10

    def __init__(self, chunks: Sequence[TextChunk]) -> None:
        super().__init__(chunks)

        self._concatenated_chunks = self._concatenate(self._chunks)

    def append(self, chunk: TextChunk) -> None:
        super().append(chunk)

        self._concatenated_chunks = self._concatenate([self._concatenated_chunks, chunk])

    def _concat_on_overlap(self, chunk1: TextChunk, chunk2: TextChunk) -> TextChunk:
        for size in reversed(range(min(len(chunk2), self._MAX_OVERLAP))):
            if chunk1._clean_text.endswith(chunk2._clean_text[:size]):
                break

        return TextChunk(chunk1.text[: len(chunk1) - size] + chunk2.text[1:])

    def _concatenate(self, chunks: list[TextChunk]) -> TextChunk:
        return reduce(self._concat_on_overlap, chunks)

    @property
    def text(self) -> str:
        return self._concatenated_chunks.text

    def __repr__(self) -> str:
        return f"ConcatenatedText: {self.text}"
