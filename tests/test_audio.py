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
    return AudioBatch.fake(nsamples=NSAMPLES, nchannels=1, sample_format=miniaudio.SampleFormat.SIGNED16)


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

    audio1ch.wav(file_path)

    with open(file_path, "rb") as f:
        bytes = f.read()

    audio_from_bytes = Audio.from_bytes(bytes, nchannels=audio1ch.nchannels, sample_rate=audio1ch.sample_rate)

    np.testing.assert_array_equal(audio_from_bytes.numpy(), audio1ch.numpy())


def test_from_file(audio1ch, tmp_path):
    file_path = str(tmp_path / "test_audio.wav")

    audio1ch.wav(file_path)

    audio_from_file = Audio.from_file(file_path, nchannels=audio1ch.nchannels, sample_rate=audio1ch.sample_rate)

    np.testing.assert_array_equal(audio_from_file.numpy(), audio1ch.numpy())


def test_from_numpy(audio1ch, audio1ch_32bit, audio2ch):
    audio_from_numpy = Audio.from_numpy(audio1ch.numpy())
    audio_from_numpy_32bit = Audio.from_numpy(audio1ch_32bit.numpy())
    audio_from_numpy_2ch = Audio.from_numpy(audio2ch.numpy(), nchannels=2)

    np.testing.assert_array_equal(audio_from_numpy.numpy(), audio1ch.numpy())
    np.testing.assert_array_equal(audio_from_numpy_32bit.numpy(), audio1ch_32bit.numpy())
    np.testing.assert_array_equal(audio_from_numpy_2ch.numpy(), audio2ch.numpy())


def test_eq(audio1ch, audio1ch_32bit, audio2ch):
    assert audio1ch == audio1ch
    assert audio1ch != audio2ch
    assert audio1ch != audio1ch_32bit


def test_add(audio1ch, audio2ch):
    audio_concat = audio1ch + audio1ch

    assert audio_concat.numpy().shape == (NSAMPLES * 2,)

    np.testing.assert_array_equal(audio_concat.numpy(), np.concatenate([audio1ch.numpy(), audio1ch.numpy()]))

    audio_concat._check_properties_equality(audio1ch)
    with pytest.raises(ValueError):
        audio1ch + audio2ch


def test_slice(audio1ch):
    audio_slice = audio1ch[1000:2000]

    assert audio_slice.numpy().shape == (1000,)
    audio_slice._check_properties_equality(audio1ch)
    np.testing.assert_array_equal(audio_slice.numpy(), audio1ch.numpy()[1000:2000])


def test_slice_2ch(audio2ch):
    audio_slice = audio2ch[1000:2000]

    assert audio_slice.numpy().shape == (2, 1000)
    audio_slice._check_properties_equality(audio2ch)
    np.testing.assert_array_equal(audio_slice.numpy(), audio2ch.numpy()[..., 1000:2000])


def test_normalization(audio1ch):
    assert audio1ch.numpy(normalize=True).mean() == pytest.approx(0.0, abs=1e-2)


def test_normalization_32bit(audio1ch_32bit):
    assert audio1ch_32bit.numpy(normalize=True).mean() == pytest.approx(0.0, abs=1e-2)


def test_batch_len(audio_batch):
    assert len(audio_batch) == 2


def test_batch_prop(audio_batch):
    for audio in audio_batch:
        assert audio.num_samples == NSAMPLES
        assert audio.nchannels == 1
        assert audio.sample_rate == SAMPLE_RATE


def test_batch_check_properties_equality(audio1ch, audio2ch):
    with pytest.raises(ValueError):
        AudioBatch([audio1ch, audio2ch])
