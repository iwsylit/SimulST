from typing import Any, Iterator, Self

import miniaudio
import numpy as np


class Audio:
    _SAMPLE_FORMAT = miniaudio.SampleFormat.SIGNED16
    _SAMPLE_RATE = 16000
    _NCHANNELS = 1

    def __init__(self, audio: miniaudio.DecodedSoundFile, normalize: bool = True) -> None:
        self._audio = audio
        self._samples = np.array(self._audio.samples, dtype=np.float32)

        if normalize:
            self._normalize()

    @classmethod
    def from_bytes(
        cls,
        bytes: bytes,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
        normalize: bool = True,
    ) -> Self:
        audio = miniaudio.decode(
            bytes,
            nchannels=nchannels,
            sample_rate=sample_rate,
            output_format=sample_format,
        )

        return cls(audio, normalize)

    @classmethod
    def from_file(
        cls,
        filename: str,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
        normalize: bool = True,
    ) -> Self:
        audio = miniaudio.decode_file(
            filename,
            nchannels=nchannels,
            sample_rate=sample_rate,
            output_format=sample_format,
        )

        return cls(audio, normalize)

    @classmethod
    def fake(
        cls,
        nsamples: int,
        nchannels: int = _NCHANNELS,
        sample_rate: int = _SAMPLE_RATE,
        sample_format: miniaudio.SampleFormat = _SAMPLE_FORMAT,
        normalize: bool = True,
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

        return cls(audio, normalize)

    def to_wav(self, filename: str) -> None:
        miniaudio.wav_write_file(filename, self._audio)

    def _normalize(self) -> None:
        norm = {
            miniaudio.SampleFormat.SIGNED16: 1 << 15,
            miniaudio.SampleFormat.SIGNED24: 1 << 23,
            miniaudio.SampleFormat.SIGNED32: 1 << 31,
        }

        if self._audio.sample_format not in norm:
            raise ValueError(f"Unsupported sample format: {self._audio.sample_format}")

        self._samples /= norm[self._audio.sample_format]

    @property
    def samples(self) -> np.ndarray:
        return self._samples

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


class AudioBatch:
    def __init__(self, audios: list[Audio]) -> None:
        self._audios = audios

    @classmethod
    def from_bytes(cls, bytes_list: list[bytes], **kwargs: Any) -> Self:
        audios = [Audio.from_bytes(b, **kwargs) for b in bytes_list]

        return cls(audios)

    @classmethod
    def from_files(cls, filenames: list[str], **kwargs: Any) -> Self:
        audios = [Audio.from_file(f, **kwargs) for f in filenames]

        return cls(audios)

    @classmethod
    def fake(cls, batch_size: int, nsamples: int, **kwargs: Any) -> Self:
        audios = [Audio.fake(nsamples, **kwargs) for _ in range(batch_size)]

        return cls(audios)

    @property
    def samples(self) -> list[np.ndarray]:
        return [audio.samples for audio in self._audios]

    def __len__(self) -> int:
        return len(self._audios)

    def __getitem__(self, index: int) -> Audio:
        return self._audios[index]

    def __iter__(self) -> Iterator[Audio]:
        return iter(self._audios)
