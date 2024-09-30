from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES as whisper_langs

from simulst.audio import Audio, AudioBatch
from simulst.translation import (
    SpeechTranscription,
    SpeechTranscriptionBatch,
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
    def _load_processor(self) -> nn.Module:
        pass

    def _check_supported_languages(self, source_lang: str | None, target_lang: str | None) -> None:
        if source_lang and source_lang not in self._SUPPORTED_SOURCE_LANGUAGES:
            raise ValueError(
                f"Source language {source_lang} is not supported.\nSupported languages: {self._SUPPORTED_SOURCE_LANGUAGES}."  # noqa: E501
            )

        if target_lang and target_lang not in self._SUPPORTED_TARGET_LANGUAGES:
            raise ValueError(
                f"Target language {target_lang} is not supported.\nSupported languages: {self._SUPPORTED_TARGET_LANGUAGES}."  # noqa: E501
            )

    @property
    def generation_params(self) -> dict:
        return self._generation_params


class AsrModel(BaseModel):
    @abstractmethod
    def _generate(
        self, audio: Audio | AudioBatch, language: str, prev_transcript: SpeechTranscription | None = None
    ) -> list[str]:
        pass

    def transcribe_batch(self, audios: AudioBatch, language: str) -> SpeechTranscriptionBatch:
        self._check_supported_languages(language, None)

        transcriptions = self._generate(audios, language)

        return SpeechTranscriptionBatch(
            [SpeechTranscription(audio, transcription) for audio, transcription in zip(audios, transcriptions)]
        )

    def transcribe(
        self, audio: Audio, language: str, prev_transcript: SpeechTranscription | None = None
    ) -> SpeechTranscription:
        self._check_supported_languages(language, None)

        transcription = self._generate(audio, language, prev_transcript)[0]

        return SpeechTranscription(audio, transcription)


class TranslationModel(BaseModel):
    @abstractmethod
    def translate_batch(
        self, texts: SpeechTranscriptionBatch, source_lang: str, target_lang: str
    ) -> TextTranslationBatch:
        pass

    def translate(self, text: SpeechTranscription, source_lang: str, target_lang: str) -> TextTranslation:
        return self.translate_batch(SpeechTranscriptionBatch([text]), source_lang, target_lang)[0]


class E2EModel(BaseModel):
    @abstractmethod
    def translate_batch(self, audios: AudioBatch, source_lang: str, target_lang: str) -> SpeechTranslationBatch:
        pass

    def translate(self, audio: Audio, source_lang: str, target_lang: str) -> SpeechTranslation:
        return self.translate_batch(AudioBatch([audio]), source_lang, target_lang)[0]


class WhisperModel(AsrModel):
    _SUPPORTED_SOURCE_LANGUAGES = set(whisper_langs.keys())
    _SUPPORTED_TARGET_LANGUAGES = {"en"}

    def __init__(self, name_or_path: str = "openai/whisper-base", generation_params: dict = {}) -> None:
        self._generation_params = {
            "prompt_condition_type": "all-segments",
            "condition_on_prev_tokens": True,
            "compression_ratio_threshold": 1.35,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "num_beams": 5,
            "temperature": 0.0,
            "max_new_tokens": 32,
        } | generation_params

        super().__init__(name_or_path, generation_params)

    @classmethod
    def fake(cls, generation_params: dict = {}) -> Self:
        class FakeWhisperModel(cls):  # type: ignore
            def _load_model(self) -> nn.Module:
                return nn.Linear(1, 1)

            def _load_processor(self) -> nn.Module:
                return nn.Linear(1, 1)

            def _generate(
                self, audio: Audio | AudioBatch, language: str, prev_transcript: SpeechTranscription | None = None
            ) -> list[str]:
                if isinstance(audio, Audio):
                    return ["fake"]
                else:
                    return ["fake" for _ in audio]

        return FakeWhisperModel("", generation_params)

    def _load_model(self) -> nn.Module:
        return WhisperForConditionalGeneration.from_pretrained(self._name_or_path)

    def _load_processor(self) -> nn.Module:
        return WhisperProcessor.from_pretrained(self._name_or_path)

    @torch.inference_mode()
    def _generate(
        self, audio: Audio | AudioBatch, language: str, prev_transcript: SpeechTranscription | None = None
    ) -> list[str]:
        # fmt: off
        input_features = self._processor(audio.numpy(normalize=True), sampling_rate=audio.sample_rate, return_tensors="pt")  # noqa: E501
        prompt_ids = self._processor.get_prompt_ids(prev_transcript.target, return_tensors="pt") if prev_transcript else None  # noqa: E501

        forced_decoder_ids = self._processor.get_decoder_prompt_ids(language=language, task="transcribe")
        # fmt: on

        predicted_ids = self._model.generate(
            input_features.input_features,
            forced_decoder_ids=forced_decoder_ids,
            prompt_ids=prompt_ids,
            **self._generation_params,
        )

        return list(map(str.strip, self._processor.batch_decode(predicted_ids, skip_special_tokens=True)))
