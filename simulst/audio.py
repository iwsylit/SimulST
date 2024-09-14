from typing import Self

import miniaudio
import numpy as np


class Audio:
    _SAMPLE_FORMAT = miniaudio.SampleFormat.SIGNED16
    _SAMPLE_RATE = 16000
    _NCHANNELS = 1

    def __init__(self, audio: miniaudio.DecodedSoundFile) -> None:
        self._audio = audio

    @classmethod
    def from_bytes(
        cls,
        bytes: bytes,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
    ) -> Self:
        audio = miniaudio.decode(
            bytes,
            nchannels=nchannels,
            sample_rate=sample_rate,
            output_format=sample_format,
        )

        return cls(audio)

    @classmethod
    def from_file(
        cls,
        filename: str,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
    ) -> Self:
        audio = miniaudio.decode_file(
            filename,
            nchannels=nchannels,
            sample_rate=sample_rate,
            output_format=sample_format,
        )

        return cls(audio)

    @classmethod
    def fake(
        cls,
        nsamples: int,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
    ) -> Self:
        samples = miniaudio._array_proto_from_format(sample_format)
        samples.frombytes(np.random.randint(0, 100, nsamples * nchannels, dtype=np.int16).tobytes())

        audio = miniaudio.DecodedSoundFile(
            name="",
            nchannels=nchannels,
            sample_rate=sample_rate,
            samples=samples,
            sample_format=sample_format,
        )

        return cls(audio)

    def save(self, filename: str) -> None:
        miniaudio.wav_write_file(filename, self._audio)

    @property
    def samples(self, normalize: bool = True) -> np.ndarray:
        audio = np.array(self._audio.samples, dtype=np.float32)

        if normalize:
            audio /= 1 << 15

        return audio

    @property
    def nchannels(self) -> int:
        return self._audio.nchannels

    @property
    def duration(self) -> float:
        return self._audio.duration

    @property
    def sample_rate(self) -> int:
        return self._audio.sample_rate

    @property
    def num_frames(self) -> int:
        return self._audio.num_frames

    @property
    def num_samples(self) -> int:
        return len(self._audio.samples)
