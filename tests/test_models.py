# mypy: allow-untyped-defs

import pytest

from simulst.audio import Audio, AudioBatch
from simulst.models import WhisperModel


@pytest.fixture
def whisper_model():
    return WhisperModel.fake()


@pytest.fixture
def audio():
    return Audio.fake()


@pytest.fixture
def audio_batch():
    return AudioBatch.fake()


@pytest.fixture
def true_whisper():
    return WhisperModel.from_pretrained("openai/whisper-tiny")


def test_transcribe_batch(whisper_model, audio_batch):
    transcription_batch = whisper_model.transcribe_batch(audio_batch, "ru")

    assert len(transcription_batch) == len(audio_batch)


def test_transcribe_eq(audio, whisper_model):
    assert whisper_model.transcribe(audio, "en") == whisper_model.transcribe(audio, "en")


def test_transcribe_error(whisper_model, audio):
    with pytest.raises(ValueError):
        whisper_model.transcribe(audio, "non_existent_language")


def test_translation_eq(whisper_model, audio):
    assert whisper_model.translate(audio, "ru", "en") == whisper_model.translate(audio, "ru", "en")


def test_translation_error(whisper_model, audio):
    with pytest.raises(ValueError):
        whisper_model.translate(audio, "ru", "non_existent_language")


def test_translate_batch(whisper_model, audio_batch):
    translation_batch = whisper_model.translate_batch(audio_batch, "ru", "en")

    assert len(translation_batch) == len(audio_batch)


@pytest.mark.slow
def test_prompt_condition(true_whisper, audio):
    # TODO: create true tests
    try:
        transcription = true_whisper.transcribe_conditioned(audio, "ru", None)
        transcription = true_whisper.transcribe_conditioned(audio, "ru", transcription)
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")
