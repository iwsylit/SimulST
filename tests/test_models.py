# mypy: allow-untyped-defs

import pytest

from simulst.audio import Audio, AudioBatch
from simulst.models import WhisperModel
from simulst.translation import SpeechTranscription


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
    return WhisperModel("openai/whisper-tiny")


@pytest.fixture
def true_audio():
    return Audio.from_file("data/test/test_ru.wav")


def test_transcribe_batch(whisper_model, audio_batch):
    transcription_batch = whisper_model.transcribe_batch(audio_batch, "ru")

    assert len(transcription_batch) == len(audio_batch)


def test_transcribe_eq(audio, whisper_model):
    assert whisper_model.transcribe(audio, "en") == whisper_model.transcribe(audio, "en")


def test_transcribe_error(whisper_model, audio):
    with pytest.raises(ValueError):
        whisper_model.transcribe(audio, "non_existent_language")


def test_generation_params():
    whisper_model = WhisperModel.fake(generation_params={"num_beams": 1, "temperature": 100.0})

    assert whisper_model.generation_params["num_beams"] == 1
    assert whisper_model.generation_params["temperature"] == 100.0


@pytest.mark.slow
def test_ru_transcribe(true_whisper, true_audio):
    transcription = true_whisper.transcribe(true_audio, "ru")

    assert transcription == SpeechTranscription(true_audio, target="тестовая запись на русском языке.")


@pytest.mark.slow
def test_ru_transcribe_with_prompt(true_whisper, true_audio):
    transcription = true_whisper.transcribe(
        true_audio, "ru", SpeechTranscription(true_audio, target="Тестовая запись на")
    )

    assert transcription == SpeechTranscription(true_audio, target="русском языке.")
