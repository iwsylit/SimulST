import queue
from abc import ABC, abstractmethod
from typing import Any

from streamlit_webrtc import WebRtcMode, webrtc_streamer

from simulst.audio import Audio
from simulst.models import AsrModel
from simulst.text_chunks import ConcatenatedText, TextChunk


class AudioStream(ABC):
    def __init__(self, chunk_size: float, buffer_size: float, sample_rate: int) -> None:
        """
        :param chunk_size: The size of the chunk in seconds.
        :param buffer_size: The buffer size in seconds.
        :param sample_rate: The sample rate of the audio.
        """
        self._chunk_size = chunk_size
        self._buffer_size = buffer_size
        self._sample_rate = sample_rate

        self._chunk_size_samples = int(self._chunk_size * self._sample_rate)
        self._buffer_size_samples = int(self._buffer_size * self._sample_rate)

        self._audio = Audio.empty(1, sample_rate=sample_rate)
        self._text = ConcatenatedText.empty()
        self._running = False

    @abstractmethod
    def process_audio(self, audio: Audio) -> TextChunk:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @property
    def text(self) -> str:
        return self._text.text

    @property
    def running(self) -> bool:
        return self._running


class AsrStream(AudioStream):
    def __init__(
        self,
        model: AsrModel,
        language: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._model = model
        self._language = language

    def process_audio(self, audio: Audio) -> TextChunk:
        return TextChunk.from_translation(self._model.transcribe(audio, language=self._language))  # type: ignore


class StreamlitWebRtcAsrStream(AsrStream):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={"video": False, "audio": True},
            audio_receiver_size=512,
        )

        self._running = False

    def run(self) -> None:
        buffer = Audio.empty(1, sample_rate=self._sample_rate)
        chunk = Audio.empty(1, sample_rate=self._sample_rate)

        if not self._webrtc_ctx.state.playing:
            return

        while True:
            if self._webrtc_ctx.audio_receiver:
                try:
                    audio_frames = self._webrtc_ctx.audio_receiver.get_frames(timeout=1)  # type: ignore
                    self._running = True
                except queue.Empty:
                    print("Stop stream.")
                    self._running = False
                    break

                for audio_frame in audio_frames:
                    chunk += Audio.from_av_frame(audio_frame).convert(1, self._sample_rate)

                if chunk.duration >= self._chunk_size:
                    buffer += chunk
                    buffer = buffer[-self._buffer_size_samples :]

                    print("Send buffer to model", buffer)
                    text_chunk = self.process_audio(buffer)
                    print("Received text chunk", text_chunk)
                    self._text.append(text_chunk)
                    self._audio += chunk

                    chunk = Audio.empty(1, sample_rate=self._sample_rate)
            else:
                print("Stop stream.")
                self._running = False
                break

        import time

        if self._audio.duration > 2.0:
            self._audio.wav(f"output {time.time()}.wav")
