from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from simulst.audio import Audio, AudioBatch
from simulst.transcription import (
    AudioTranscription,
    AudioTranscriptionBatch,
    SpeechTranslation,
    TextTranslation,
)


class BaseModel(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def from_pretrained(cls, path: str) -> Self:
        pass


class AsrModel(BaseModel):
    @abstractmethod
    def transcribe_batch(self, audios: AudioBatch, language: str) -> AudioTranscriptionBatch:
        pass

    def transcribe(self, audio: Audio, language: str) -> AudioTranscription:
        return self.transcribe_batch(AudioBatch([audio]), language)[0]


class TranslationModel(BaseModel):
    @abstractmethod
    def translate(self, text: AudioTranscription) -> TextTranslation:
        pass


class E2EModel(BaseModel):
    @abstractmethod
    def translate(self, audio: Audio) -> SpeechTranslation:
        pass


class WhisperModel(AsrModel):
    def __init__(self, path: str = "openai/whisper-base", task: str = "transcribe") -> None:
        super().__init__()

        self.model = WhisperForConditionalGeneration.from_pretrained(path)
        self.processor = WhisperProcessor.from_pretrained(path)

        self._task = task

    @classmethod
    def from_pretrained(cls, path: str = "openai/whisper-base") -> Self:
        return cls(path, "transcribe")

    @torch.inference_mode()
    def transcribe_batch(self, audios: AudioBatch, language: str) -> AudioTranscriptionBatch:
        input_features = self.processor(audios.samples, sampling_rate=audios.sample_rate, return_tensors="pt")
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=self._task)

        predicted_ids = self.model.generate(input_features.input_features, forced_decoder_ids=forced_decoder_ids)
        transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return AudioTranscriptionBatch(
            [AudioTranscription(audio, transcription) for audio, transcription in zip(audios, transcriptions)]
        )
