# mypy: allow-untyped-defs

import miniaudio
import numpy as np
import pytest

from simulst.audio import Audio, AudioBatch

SAMPLE_RATE = 16000
NSAMPLES = SAMPLE_RATE * 4


@pytest.fixture
def audio1ch():
    return Audio.fake(NSAMPLES, 1, SAMPLE_RATE)


@pytest.fixture
def audio2ch():
    return Audio.fake(NSAMPLES, 2, SAMPLE_RATE)


@pytest.fixture
def audio1ch_32bit():
    return Audio.fake(NSAMPLES, 1, SAMPLE_RATE, miniaudio.SampleFormat.SIGNED32)


@pytest.fixture
def audio_batch():
    return AudioBatch.fake(2, NSAMPLES, nchannels=1, sample_format=miniaudio.SampleFormat.SIGNED16)


def test_duration(audio1ch, audio2ch):
    assert audio1ch.duration == audio2ch.duration


def test_duration_1ch(audio1ch):
    assert audio1ch.duration == 4.0


def test_duration_2ch(audio2ch):
    assert audio2ch.duration == 4.0


def test_sample_rate(audio1ch, audio2ch):
    assert audio1ch.sample_rate == audio2ch.sample_rate


def test_num_samples(audio1ch, audio2ch):
    assert audio1ch.num_samples * 2 == audio2ch.num_samples


def test_num_frames(audio1ch, audio2ch):
    assert audio1ch.num_frames == audio2ch.num_frames


def test_from_bytes(audio1ch, tmp_path):
    file_path = str(tmp_path / "test_audio.wav")

    audio1ch.to_wav(file_path)

    with open(file_path, "rb") as f:
        bytes = f.read()

    audio_from_bytes = Audio.from_bytes(bytes, nchannels=audio1ch.nchannels, sample_rate=audio1ch.sample_rate)

    np.testing.assert_array_equal(audio_from_bytes.samples, audio1ch.samples)


def test_from_file(audio1ch, tmp_path):
    file_path = str(tmp_path / "test_audio.wav")

    audio1ch.to_wav(file_path)

    audio_from_file = Audio.from_file(file_path, nchannels=audio1ch.nchannels, sample_rate=audio1ch.sample_rate)

    np.testing.assert_array_equal(audio_from_file.samples, audio1ch.samples)


def test_normalization(audio1ch):
    assert audio1ch.samples.mean() == pytest.approx(0.0, abs=1e-2)


def test_normalization_32bit(audio1ch_32bit):
    assert audio1ch_32bit.samples.mean() == pytest.approx(0.0, abs=1e-2)


def test_batch_len(audio_batch):
    assert len(audio_batch) == 2


def test_batch_num_samples(audio_batch):
    for audio in audio_batch:
        assert audio.num_samples == NSAMPLES


def test_batch_num_channels(audio_batch):
    for audio in audio_batch:
        assert audio.nchannels == 1


def test_batch_sample_rate(audio_batch):
    for audio in audio_batch:
        assert audio.sample_rate == SAMPLE_RATE
