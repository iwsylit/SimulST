import re
from functools import reduce
from typing import Iterator, Self, Sequence

from simulst.translation import Translation


class TextChunk:
    _CLEAN_TEXT_REGEX = re.compile(r"[^\w\s]")

    def __init__(self, text: str) -> None:
        self._text = text

    @classmethod
    def from_translation(cls, translation: Translation) -> Self:
        return cls(translation.target)

    def _clean(self) -> str:
        return self._CLEAN_TEXT_REGEX.sub("", self.text).lower().strip()

    def _decapitalize(self) -> str:
        return self.text[0].lower() + self.text[1:]

    @property
    def text(self) -> str:
        return self._text

    @property
    def clean_text(self) -> str:
        return self._clean()

    @property
    def decapitalized_text(self) -> str:
        return self._decapitalize()

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

    def __len__(self) -> int:
        return len(self._chunks)


class ConcatenatedText(TextChunks):
    _CLEAN_END_REGEX = re.compile(r"[^\w]+$")
    _MAX_OVERLAP = 10

    def __init__(self, chunks: Sequence[TextChunk]) -> None:
        super().__init__(chunks)

        self._concatenated_chunks = self._concatenate(self._chunks)

    def append(self, chunk: TextChunk) -> None:
        super().append(chunk)

        self._concatenated_chunks = self._concatenate([self._concatenated_chunks, chunk])

    def _concat_on_overlap(self, chunk1: TextChunk, chunk2: TextChunk) -> TextChunk:
        for size in reversed(range(min(len(chunk2), self._MAX_OVERLAP))):
            if chunk1.clean_text.endswith(chunk2.clean_text[:size]):
                break

        if size == 0:
            return TextChunk(chunk1.text + " " + chunk2.decapitalized_text)

        return TextChunk(self._CLEAN_END_REGEX.sub("", chunk1.text) + chunk2.text[size:])

    def _concatenate(self, chunks: list[TextChunk]) -> TextChunk:
        return reduce(self._concat_on_overlap, chunks)

    @property
    def text(self) -> str:
        return self._concatenated_chunks.text

    def __repr__(self) -> str:
        return f"ConcatenatedText: {self.text}"
