from typing import Generic, Self, Sequence, TypeVar

from simulst.audio import Audio

T = TypeVar("T", Audio, str)


class Transcription(Generic[T]):
    def __init__(self, source: T, target: str) -> None:
        self._source: T = source
        self._target = target

    @classmethod
    def fake(cls) -> Self:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def source(self) -> T:
        return self._source

    @property
    def target(self) -> str:
        return self._target

    def __repr__(self) -> str:
        return f"Source: [{self.source}], Translation: {self.target}"

    def __eq__(self, other: Self) -> bool:  # type: ignore
        return self.target == other.target


class Translation(Transcription, Generic[T]):
    def __init__(self, source: T, target: str, source_lang: str, target_lang: str) -> None:
        super().__init__(source, target)

        self._source_lang = source_lang
        self._target_lang = target_lang

    @property
    def source_lang(self) -> str:
        return self._source_lang

    @property
    def target_lang(self) -> str:
        return self._target_lang


class SpeechTranscription(Transcription[Audio]):
    @classmethod
    def fake(cls) -> Self:
        return cls(Audio.fake(), "fake transcription")


class SpeechTranslation(Translation[Audio]):
    @classmethod
    def fake(cls) -> Self:
        return cls(Audio.fake(), "fake translation", "fake source lang", "fake target lang")


class TextTranslation(Translation[str]):
    @classmethod
    def fake(cls) -> Self:
        return cls("fake source text", "fake translation", "fake source lang", "fake target lang")


B = TypeVar("B", SpeechTranscription, SpeechTranslation, TextTranslation)


class TranscriptionBatch(Generic[B]):
    def __init__(self, items: Sequence[B]) -> None:
        self._items: list[B] = list(items)

    @classmethod
    def fake(cls, batch_size: int = 2) -> Self:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def target(self) -> list[str]:
        return [item.target for item in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> B:
        return self._items[index]

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if len(self) <= 10:
            return f"{class_name}:\n- " + "\n- ".join([repr(t) for t in self._items])
        else:
            return f"{class_name} containing {len(self)} items"


class SpeechTranscriptionBatch(TranscriptionBatch[SpeechTranscription]):
    @classmethod
    def fake(cls, batch_size: int = 2) -> Self:
        return cls([SpeechTranscription.fake() for _ in range(batch_size)])


class SpeechTranslationBatch(TranscriptionBatch[SpeechTranslation]):
    @classmethod
    def fake(cls, batch_size: int = 2) -> Self:
        return cls([SpeechTranslation.fake() for _ in range(batch_size)])


class TextTranslationBatch(TranscriptionBatch[TextTranslation]):
    @classmethod
    def fake(cls, batch_size: int = 2) -> Self:
        return cls([TextTranslation.fake() for _ in range(batch_size)])
