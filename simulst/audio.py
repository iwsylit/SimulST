import array
from typing import Any, Callable, Iterator, Self, Sequence

import miniaudio
import numpy as np


def _numpy_to_array_typecode(dtype: np.dtype) -> str:
    if dtype == np.int16:
        return "h"
    elif dtype == np.int32:
        return "i"
    elif dtype == np.float32:
        return "f"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


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
    def _create(
        cls,
        factory_method: Callable,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
        **kwargs: Any,
    ) -> Self:
        audio = factory_method(nchannels=nchannels, sample_rate=sample_rate, output_format=sample_format, **kwargs)

        return cls(audio)

    @classmethod
    def from_bytes(cls, bytes: bytes, **kwargs: Any) -> Self:
        """
        :param bytes: The bytes to decode.
        :param kwargs: Additional arguments to pass to the miniaudio.decode function.
        """
        return cls._create(miniaudio.decode, data=bytes, **kwargs)

    @classmethod
    def from_file(cls, filename: str, **kwargs: Any) -> Self:
        """
        :param filename: The filename to decode.
        :param kwargs: Additional arguments to pass to the miniaudio.decode_file function.
        """
        return cls._create(miniaudio.decode_file, filename=filename, **kwargs)

    @classmethod
    def from_numpy(
        cls,
        samples: np.ndarray,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
    ) -> Self:
        audio = miniaudio.DecodedSoundFile(
            name="",
            nchannels=nchannels,
            sample_rate=sample_rate,
            sample_format=sample_format,
            samples=array.array(_numpy_to_array_typecode(samples.dtype), samples.tobytes()),
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
        if self.nchannels > 1:
            samples = samples.reshape(self.nchannels, -1)

        if normalize:
            if self._audio.sample_format not in self._NORM:
                raise ValueError(f"Unsupported sample format: {self._audio.sample_format}")

            return samples / self._NORM[self._audio.sample_format]

        return samples

    def _equal_properties(self, other: Self) -> bool:
        return (
            self.nchannels == other.nchannels
            and self.sample_rate == other.sample_rate
            and self.sample_format == other.sample_format
        )

    def _check_properties_equality(self, other: Self) -> None:
        if not self._equal_properties(other):
            raise ValueError("Both audios must have the same number of channels, sample rate and sample format")

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

    @property
    def sample_format(self) -> miniaudio.SampleFormat:
        return self._audio.sample_format

    def __repr__(self) -> str:
        return f"Audio: {self.duration:.2f}s ({self.nchannels}ch, {self.sample_rate}Hz)"

    def __eq__(self, other: Self) -> bool:  # type: ignore
        if not self._equal_properties(other):
            return False

        return np.array_equal(self.numpy(), other.numpy())

    def __add__(self, other: Self) -> Self:
        self._check_properties_equality(other)

        return Audio.from_numpy(
            np.concatenate([self.numpy(), other.numpy()]),
            nchannels=self.nchannels,
            sample_rate=self.sample_rate,
            sample_format=self._audio.sample_format,
        )  # type: ignore

    def __getitem__(self, slice: slice) -> Self:
        return Audio.from_numpy(
            self.numpy()[..., slice],
            nchannels=self.nchannels,
            sample_rate=self.sample_rate,
            sample_format=self._audio.sample_format,
        )  # type: ignore


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
        if not all(self._audios[0]._equal_properties(audio) for audio in self._audios):
            raise ValueError("All audios must have the same number of channels, sample rate and sample format")

    def __len__(self) -> int:
        return len(self._audios)

    def __getitem__(self, index: int) -> Audio:
        return self._audios[index]

    def __iter__(self) -> Iterator[Audio]:
        return iter(self._audios)
