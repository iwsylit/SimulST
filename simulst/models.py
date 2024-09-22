from abc import ABC, abstractmethod
from typing import Literal, Self

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

# TODO: make this class ok and fix the class structure


class BaseModel(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def from_pretrained(cls, path: str) -> Self:
        pass


class AsrModel(BaseModel):
    @abstractmethod
    def transcribe_batch(self, audios: AudioBatch, language: str) -> SpeechTranscriptionBatch:
        pass

    def transcribe(self, audio: Audio, language: str) -> SpeechTranscription:
        return self.transcribe_batch(AudioBatch([audio]), language)[0]


class TranslationModel(BaseModel):
    @abstractmethod
    def translate_batch(
        self, texts: SpeechTranscriptionBatch, source_lang: str, target_lang: str
    ) -> TextTranslationBatch:
        pass

    def translate(self, text: SpeechTranscription, source_lang: str, target_lang: str) -> TextTranslation:
        return self.translate_batch(SpeechTranscriptionBatch([text]), source_lang, target_lang)[0]


class E2EModel(BaseModel):
    _SUPPORTED_SOURCE_LANGUAGES: list[str] = []
    _SUPPORTED_TARGET_LANGUAGES: list[str] = []

    @abstractmethod
    def translate_batch(self, audios: AudioBatch, source_lang: str, target_lang: str) -> SpeechTranslationBatch:
        pass

    def translate(self, audio: Audio, source_lang: str, target_lang: str) -> SpeechTranslation:
        return self.translate_batch(AudioBatch([audio]), source_lang, target_lang)[0]

    def _check_supported_languages(self, source_lang: str | None, target_lang: str | None) -> None:
        if source_lang and source_lang not in self._SUPPORTED_SOURCE_LANGUAGES:
            raise ValueError(
                f"Source language {source_lang} is not supported.\nSupported languages: {self._SUPPORTED_SOURCE_LANGUAGES}."  # noqa: E501
            )

        if target_lang and target_lang not in self._SUPPORTED_TARGET_LANGUAGES:
            raise ValueError(
                f"Target language {target_lang} is not supported.\nSupported languages: {self._SUPPORTED_TARGET_LANGUAGES}."  # noqa: E501
            )


class WhisperModel(AsrModel, E2EModel):

    _SUPPORTED_SOURCE_LANGUAGES = list(whisper_langs.keys())
    _SUPPORTED_TARGET_LANGUAGES = ["en"]

    def __init__(self, path: str = "openai/whisper-base") -> None:
        super().__init__()

        self.model = WhisperForConditionalGeneration.from_pretrained(path)
        self.processor = WhisperProcessor.from_pretrained(path)
        self._generation_params = {
            "prompt_condition_type": "all-segments",
            "condition_on_prev_tokens": True,
            "compression_ratio_threshold": 1.35,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "num_beams": 5,
            "temperature": 0.0,
        }

    @classmethod
    def from_pretrained(cls, path: str = "openai/whisper-base") -> Self:
        return cls(path)

    @classmethod
    def fake(cls) -> Self:
        class FakeWhisperModel(cls):  # type: ignore
            def __init__(self, path: str = "openai/whisper-base") -> None:
                BaseModel.__init__(self)
                self.fake_module = nn.Linear(1, 1)

            def _run_model(
                self, audios: AudioBatch, language: str, task: Literal["transcribe", "translate"]
            ) -> list[str]:
                return [f"fake {task}" for _ in audios]

        return FakeWhisperModel()

    @torch.inference_mode()
    def _run_model(
        self,
        audios: AudioBatch,
        language: str,
        task: Literal["transcribe", "translate"],
        prev_transcript: SpeechTranscriptionBatch | None = None,
    ) -> list[str]:
        input_features = self.processor(
            audios.numpy(normalize=True), sampling_rate=audios.sample_rate, return_tensors="pt"
        )
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)

        prompt_ids = (
            self.processor.get_prompt_ids(prev_transcript.target, return_tensors="pt") if prev_transcript else None
        )

        predicted_ids = self.model.generate(
            input_features.input_features,
            forced_decoder_ids=forced_decoder_ids,
            prompt_ids=prompt_ids,
            **self._generation_params,
        )

        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

    def transcribe_batch(self, audios: AudioBatch, language: str) -> SpeechTranscriptionBatch:
        self._check_supported_languages(language, None)

        transcriptions = self._run_model(audios, language, "transcribe")

        return SpeechTranscriptionBatch(
            [SpeechTranscription(audio, transcription) for audio, transcription in zip(audios, transcriptions)]
        )

    def translate_batch(self, audios: AudioBatch, source_lang: str, target_lang: str) -> SpeechTranslationBatch:
        self._check_supported_languages(source_lang, target_lang)

        translations = self._run_model(audios, source_lang, "translate")

        return SpeechTranslationBatch(
            [
                SpeechTranslation(audio, translation, source_lang, target_lang)
                for audio, translation in zip(audios, translations)
            ]
        )

    def transcribe_conditioned(
        self, audio: Audio, language: str, prev_transcript: SpeechTranscriptionBatch
    ) -> SpeechTranscription:
        transcription = self._run_model(AudioBatch([audio]), language, "transcribe", prev_transcript)[0]

        return SpeechTranscription(audio, transcription)
