import array
from typing import Any, Callable, Iterator, Self, Sequence

import av
import miniaudio
import numpy as np


class Audio:
    _SAMPLE_FORMAT = miniaudio.SampleFormat.SIGNED16
    _SAMPLE_RATE = 16000
    _NCHANNELS = 1

    _NORM = {
        miniaudio.SampleFormat.SIGNED16: 1 << 15,
        miniaudio.SampleFormat.SIGNED32: 1 << 31,
    }

    _NUMPY_DTYPE_MAP = {
        miniaudio.SampleFormat.SIGNED16: np.int16,
        miniaudio.SampleFormat.SIGNED32: np.int32,
    }

    _AV_FORMAT_MAP = {
        "s16": miniaudio.SampleFormat.SIGNED16,
        "s32": miniaudio.SampleFormat.SIGNED32,
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
    def from_array(cls, samples: array.array, **kwargs: Any) -> Self:
        """
        :param samples: The samples to decode.
        :param kwargs: Additional arguments to pass to the miniaudio.decode function.
        """

        def factory_method(samples: array.array, **kwargs: Any) -> miniaudio.DecodedSoundFile:
            kwargs["sample_format"] = kwargs.pop("output_format")

            return miniaudio.DecodedSoundFile(name="", samples=samples, **kwargs)

        return cls._create(factory_method, samples=samples, **kwargs)

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
        cls, samples: np.ndarray, sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT, **kwargs: Any
    ) -> Self:
        """
        :param samples: The samples to decode.
        :param kwargs: Additional arguments to pass to the miniaudio.decode function.
        """
        if samples.shape[0] == 2:  # convert 2D array with shape (nchannels, num_samples) to packed samples
            samples = samples.T.flatten()

        samples_array = miniaudio._array_proto_from_format(sample_format)
        samples_array.frombytes(samples.astype(cls._NUMPY_DTYPE_MAP[sample_format]).tobytes())

        return cls.from_array(samples_array, sample_format=sample_format, **kwargs)

    @classmethod
    def from_av_frame(cls, frame: av.AudioFrame) -> Self:
        if frame.format.name not in cls._AV_FORMAT_MAP:
            raise ValueError(f"Unsupported format: {frame.format.name}")

        return cls.from_numpy(
            frame.to_ndarray(),
            nchannels=len(frame.layout.channels),
            sample_rate=frame.sample_rate,
            sample_format=cls._AV_FORMAT_MAP[frame.format.name],
        )

    @classmethod
    def empty(cls, num_channels: int = _NCHANNELS, **kwargs: Any) -> Self:
        return cls.from_numpy(np.empty([num_channels, 0], dtype=np.float32), nchannels=num_channels, **kwargs)

    @classmethod
    def fake(
        cls,
        nsamples: int = 1000,
        nchannels: int = _NCHANNELS,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
        **kwargs: Any,
    ) -> Self:
        samples = miniaudio._array_proto_from_format(sample_format)
        samples.frombytes(np.random.randint(0, 100, nsamples * nchannels, dtype=np.int16).tobytes())

        return cls.from_array(samples, nchannels=nchannels, sample_format=sample_format, **kwargs)

    def wav(self, filename: str) -> None:
        miniaudio.wav_write_file(filename, self._audio)

    def numpy(self, normalize: bool = False) -> np.ndarray:
        samples = np.array(self._audio.samples, dtype=self._NUMPY_DTYPE_MAP[self.sample_format])
        # convert packed samples to 2D array with shape (nchannels, num_samples)
        samples = samples.reshape(-1, self.nchannels).T

        if normalize:
            samples = samples / self._NORM[self._audio.sample_format]
            return samples.astype(np.float32)

        return samples

    def convert(
        self,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
    ) -> Self:
        converted_frames = miniaudio.convert_frames(
            from_fmt=self._audio.sample_format,
            from_numchannels=self._audio.nchannels,
            from_samplerate=self._audio.sample_rate,
            sourcedata=miniaudio.ffi.from_buffer(self._audio.samples),
            to_fmt=sample_format,
            to_numchannels=nchannels,
            to_samplerate=sample_rate,
        )

        samples = miniaudio._array_proto_from_format(sample_format)
        samples.frombytes(converted_frames)

        return self.from_array(samples, nchannels=nchannels, sample_rate=sample_rate, sample_format=sample_format)

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
            np.concatenate([self.numpy(), other.numpy()], axis=1),
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
