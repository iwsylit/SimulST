from abc import ABC, abstractmethod

import whisper
from torch import nn

from simulst.audio import Audio, AudioBatch
from simulst.translation import (
    SpeechTranslation,
    SpeechTranslationBatch,
    TextTranslation,
    TextTranslationBatch,
)


class BaseModel(ABC, nn.Module):
    _SUPPORTED_SOURCE_LANGUAGES: set[str] = set()
    _SUPPORTED_TARGET_LANGUAGES: set[str] = set()

    def __init__(self, name_or_path: str, generation_params: dict = {}) -> None:
        super().__init__()

        self._name_or_path = name_or_path
        self._generation_params = generation_params

        self._processor = self._load_processor()
        self._model = self._load_model()

    @abstractmethod
    def _load_model(self) -> nn.Module:
        pass

    @abstractmethod
    def _load_processor(self) -> nn.Module | None:
        pass

    @property
    def generation_params(self) -> dict:
        return self._generation_params


class SpeechToTextModel(BaseModel):
    @abstractmethod
    def _generate(
        self,
        audio: Audio | AudioBatch,
        source_language: str,
        target_language: str,
        previous_translation: str | None = None,
    ) -> list[str]:
        pass

    def translate_batch(
        self, audios: AudioBatch, source_language: str, target_language: str
    ) -> SpeechTranslationBatch:
        transcriptions = self._generate(audios, source_language, target_language)

        return SpeechTranslationBatch(
            [
                SpeechTranslation(audio, transcription, source_language, target_language)
                for audio, transcription in zip(audios, transcriptions)
            ]
        )

    def translate(
        self, audio: Audio, source_language: str, target_language: str, previous_translation: str | None = None
    ) -> SpeechTranslation:
        transcription = self._generate(audio, source_language, target_language, previous_translation)[0]

        return SpeechTranslation(audio, transcription, source_language, target_language)


class TextToTextModel(BaseModel):
    @abstractmethod
    def translate_batch(self, texts: TextTranslationBatch, source_lang: str, target_lang: str) -> TextTranslationBatch:
        pass

    def translate(self, text: SpeechTranslation, source_lang: str, target_lang: str) -> TextTranslation:
        return self.translate_batch(SpeechTranslationBatch([text]), source_lang, target_lang)[0]


class WhisperModel(SpeechToTextModel):
    def _load_model(self) -> nn.Module:
        return whisper.load_model(self._name_or_path, device="cuda")

    def _load_processor(self) -> nn.Module:
        return None

    def _generate(
        self,
        audio: Audio | AudioBatch,
        source_language: str,
        target_language: str,
        previous_translation: str | None = None,
    ) -> list[str]:
        options = whisper.DecodingOptions(
            prefix=previous_translation,
            language=target_language,
            without_timestamps=True,
            fp16=False,
        )

        audio = whisper.pad_or_trim(audio.numpy(normalize=True).squeeze())
        mel = whisper.log_mel_spectrogram(audio, n_mels=self._model.dims.n_mels).to(self._model.device)
        output = self._model.decode(mel, options)

        return [output.text]
