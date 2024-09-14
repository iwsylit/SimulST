# mypy: allow-untyped-defs

import numpy as np
import pytest

from simulst.audio import Audio


@pytest.fixture
def audio1ch():
    sample_rate = 16000
    nchannels = 1
    nsamples = sample_rate * 4

    return Audio.fake(nsamples, nchannels, sample_rate)


@pytest.fixture
def audio2ch():
    sample_rate = 16000
    nchannels = 2
    nsamples = sample_rate * 4

    return Audio.fake(nsamples, nchannels, sample_rate)


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
