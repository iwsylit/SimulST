# mypy: allow-untyped-defs

import pytest

from simulst.audio import Audio, AudioBatch
from simulst.models import WhisperModel


@pytest.fixture
def whisper_model():
    return WhisperModel("openai/whisper-tiny")


@pytest.fixture
def audio():
    return Audio.fake()


@pytest.fixture
def audio_batch():
    return AudioBatch.fake()


def test_transcribe_batch(whisper_model, audio_batch):
    transcription_batch = whisper_model.transcribe_batch(audio_batch, "ru")

    assert len(transcription_batch) == len(audio_batch)


def test_transcribe_eq(audio, whisper_model):
    assert whisper_model.transcribe(audio, "en") == whisper_model.transcribe(audio, "en")


def test_transcribe_error(whisper_model, audio):
    with pytest.raises(ValueError):
        whisper_model.transcribe(audio, "non_existent_language")
