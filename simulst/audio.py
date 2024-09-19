import array
from typing import Any, Iterator, Self, Sequence

import miniaudio
import numpy as np


class Audio:
    _SAMPLE_FORMAT = miniaudio.SampleFormat.SIGNED16
    _SAMPLE_RATE = 16000
    _NCHANNELS = 1

    _NORM = {
        miniaudio.SampleFormat.SIGNED16: 1 << 15,
        miniaudio.SampleFormat.SIGNED24: 1 << 23,
        miniaudio.SampleFormat.SIGNED32: 1 << 31,
    }

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
    def from_numpy(
        cls,
        samples: np.ndarray,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
    ) -> Self:
        def numpy_to_array_typecode(dtype: np.dtype) -> str:
            if dtype == np.int16:
                return "h"
            elif dtype == np.int32:
                return "i"
            elif dtype == np.float32:
                return "f"
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        audio = miniaudio.DecodedSoundFile(
            name="",
            nchannels=nchannels,
            sample_rate=sample_rate,
            sample_format=sample_format,
            samples=array.array(numpy_to_array_typecode(samples.dtype), samples.tobytes()),
        )

        return cls(audio)

    @classmethod
    def fake(
        cls,
        nsamples: int = 1000,
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

    def wav(self, filename: str) -> None:
        miniaudio.wav_write_file(filename, self._audio)

    def numpy(self, normalize: bool = False) -> np.ndarray:
        samples = np.array(self._audio.samples, dtype=np.float32)

        if normalize:
            if self._audio.sample_format not in self._NORM:
                raise ValueError(f"Unsupported sample format: {self._audio.sample_format}")

            return samples / self._NORM[self._audio.sample_format]

        return samples

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

    def __repr__(self) -> str:
        return f"Audio: {self.duration:.2f}s ({self.nchannels}ch, {self.sample_rate}Hz)"


class AudioBatch:
    def __init__(self, audios: Sequence[Audio]) -> None:
        self._audios = audios

        self._check_properties_equality()

    @classmethod
    def from_bytes(cls, bytes_list: list[bytes], **kwargs: Any) -> Self:
        audios = [Audio.from_bytes(b, **kwargs) for b in bytes_list]

        return cls(audios)

    @classmethod
    def from_files(cls, filenames: list[str], **kwargs: Any) -> Self:
        audios = [Audio.from_file(f, **kwargs) for f in filenames]

        return cls(audios)

    @classmethod
    def fake(cls, batch_size: int = 2, **kwargs: Any) -> Self:
        audios = [Audio.fake(**kwargs) for _ in range(batch_size)]

        return cls(audios)

    def numpy(self, normalize: bool = False) -> list[np.ndarray]:
        return [audio.numpy(normalize) for audio in self._audios]

    @property
    def nchannels(self) -> int:
        return self._audios[0].nchannels

    @property
    def sample_rate(self) -> int:
        return self._audios[0].sample_rate

    def _check_properties_equality(self) -> None:
        for audio in self._audios:
            if audio.nchannels != self.nchannels:
                raise ValueError("All audios must have the same number of channels")

            if audio.sample_rate != self.sample_rate:
                raise ValueError("All audios must have the same sample rate")

    def __len__(self) -> int:
        return len(self._audios)

    def __getitem__(self, index: int) -> Audio:
        return self._audios[index]

    def __iter__(self) -> Iterator[Audio]:
        return iter(self._audios)
