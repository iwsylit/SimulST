from abc import ABC
from typing import Generic, Self, Sequence, TypeVar

from simulst.audio import Audio


class Translation(ABC):
    def __init__(self, source: str | Audio, translation: str, source_lang: str, target_lang: str) -> None:
        self.source = source
        self._translation = translation.strip()
        self._source_lang = source_lang
        self._target_lang = target_lang

    @property
    def translation(self) -> str:
        return self._translation

    @property
    def source_lang(self) -> str:
        return self._source_lang

    @property
    def target_lang(self) -> str:
        return self._target_lang

    def __repr__(self) -> str:
        return f"Source: [{self.source}], Translation ({self.source_lang} -> {self.target_lang}): {self.translation}"

    def __eq__(self, other: Self) -> bool:  # type: ignore
        return self.translation == other.translation and self.target_lang == other.target_lang


class TextTranslation(Translation):
    def __init__(self, source: str, translation: str, source_lang: str, target_lang: str) -> None:
        super().__init__(source, translation, source_lang, target_lang)


class SpeechTranslation(Translation):
    def __init__(self, source: Audio, translation: str, source_lang: str, target_lang: str) -> None:
        super().__init__(source, translation, source_lang, target_lang)


T = TypeVar("T", bound=Translation)


class TranslationBatch(ABC, Generic[T]):
    def __init__(self, translations: Sequence[T]) -> None:
        self._translations = translations

    def __len__(self) -> int:
        return len(self._translations)

    def __getitem__(self, index: int) -> T:
        return self._translations[index]

    def __repr__(self) -> str:
        if len(self) <= 10:
            return "TranslationBatch:\n- " + "\n- ".join([repr(t) for t in self._translations])
        else:
            return f"TranslationBatch containing {len(self)} translations"


class SpeechTranslationBatch(TranslationBatch[SpeechTranslation]):
    def __init__(self, translations: Sequence[SpeechTranslation]) -> None:
        super().__init__(translations)


class TextTranslationBatch(TranslationBatch[TextTranslation]):
    def __init__(self, translations: Sequence[TextTranslation]) -> None:
        super().__init__(translations)
