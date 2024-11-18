import datetime
import queue
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from simuleval.data.segments import SpeechSegment
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from simulst.audio import Audio
from simulst.models import SpeechToTextModel
from simulst.translation import SpeechTranscription


class AudioStream(ABC):
    TMP_DIR = Path("data/outputs")

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
        self._text = ""
        self._running = False

    @abstractmethod
    def process_audio(self, audio: Audio) -> SpeechTranscription:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def write_result(self) -> None:
        out_dir = self.TMP_DIR / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._audio.duration > 2.0:
            self._audio.wav(str(out_dir / "audio.wav"))

            with open(out_dir / "text.txt", "w") as f:
                f.write(self._text)

    @property
    def text(self) -> str:
        return self._text

    @property
    def running(self) -> bool:
        return self._running


class AsrStream(AudioStream):
    def __init__(
        self,
        model: SpeechToTextModel,
        language: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._model = model
        self._language = language
        self._states = self._model.build_states()

    def process_audio(self, audio: Audio) -> SpeechTranscription:
        segment = SpeechSegment(
            content=audio.numpy().squeeze().tolist(), sample_rate=audio.sample_rate, finished=False
        )
        output = self._model.pushpop(segment, states=self._states)
        self._states.source_finished = False

        return output.content


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
        chunk = Audio.empty(2, sample_rate=48000)

        if not self._webrtc_ctx.state.playing:
            return

        while True:
            if self._webrtc_ctx.audio_receiver:
                try:
                    audio_frames = self._webrtc_ctx.audio_receiver.get_frames(timeout=1)
                    self._running = True
                except queue.Empty:
                    print("Stop stream.")
                    self._running = False
                    break

                for audio_frame in audio_frames:
                    chunk += Audio.from_av_frame(audio_frame)

                if chunk.duration >= self._chunk_size:
                    chunk = chunk.convert(1, self._sample_rate)

                    print("Send buffer to model", chunk)
                    text_chunk = self.process_audio(chunk)
                    print("Received text chunk", text_chunk)
                    self._text += " " + text_chunk
                    print("Text", self._text)
                    self._audio += chunk

                    chunk = Audio.empty(2, sample_rate=48000)
            else:
                print("Stop stream.")
                self._running = False
                break

        self.write_result()
