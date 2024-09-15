from typing import Self, Sequence

from simulst.audio import Audio


class AudioTranscription:
    def __init__(self, audio: Audio, transcription: str) -> None:
        self._audio = audio
        self._transcription = transcription.strip()

    @classmethod
    def fake(cls) -> Self:
        return cls(Audio.fake(), "fake transcription")

    @property
    def audio(self) -> Audio:
        return self._audio

    @property
    def transcription(self) -> str:
        return self._transcription

    def __repr__(self) -> str:
        return f"Source: [{self.audio}], Transcription: {self.transcription}"

    def __eq__(self, other: Self) -> bool:  # type: ignore
        return self.transcription == other.transcription


class AudioTranscriptionBatch:
    def __init__(self, transcriptions: Sequence[AudioTranscription]) -> None:
        self._transcriptions = transcriptions

    @classmethod
    def fake(cls, batch_size: int = 2) -> Self:
        return cls([AudioTranscription.fake() for _ in range(batch_size)])

    @property
    def transcription(self) -> list[str]:
        return [t.transcription for t in self._transcriptions]

    def __len__(self) -> int:
        return len(self._transcriptions)

    def __getitem__(self, index: int) -> AudioTranscription:
        return self._transcriptions[index]

    def __repr__(self) -> str:
        if len(self) <= 10:
            return "AudioTranscriptionBatch:\n- " + "\n- ".join([repr(t) for t in self._transcriptions])
        else:
            return f"AudioTranscriptionBatch containing {len(self)} transcriptions"
