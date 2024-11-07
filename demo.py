# mypy: allow-untyped-defs

import threading
import time

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from simulst.models import WaitkWhisper
from simulst.stream import StreamlitWebRtcAsrStream

st.set_page_config(layout="wide")
st.title("Simultaneous Machine Translation")

st.sidebar.title("Language Selection")

languages = ["ru", "en", "es", "fr", "de", "zh", "ja"]

source_language = st.sidebar.selectbox("Select source language", languages)
target_language = st.sidebar.selectbox("Select target language", languages)


class WaitkWhisperConfig:
    def __init__(self, config):
        self.waitk_lagging = config.get("waitk_lagging")
        self.source_segment_size = config.get("source_segment_size")
        self.source_language = config.get("source_language")
        self.continuous_write = config.get("continuous_write")
        self.model_size = config.get("model_size")
        self.task = config.get("task")


@st.cache_resource
def load_model():
    return WaitkWhisper(
        WaitkWhisperConfig(
            {
                "waitk_lagging": 1,
                "source_segment_size": 100,
                "source_language": source_language,
                "continuous_write": 1,
                "model_size": "tiny",
                "task": "transcribe",
            }
        )
    )


whisper_model = load_model()
stream = StreamlitWebRtcAsrStream(
    model=whisper_model, language=source_language, chunk_size=1, buffer_size=1, sample_rate=16000
)


def print_text():
    while True:
        if stream.running:
            text_output.markdown(stream.text)
        time.sleep(0.2)


st.markdown("## Transcript")
text_output = st.empty()
writer = threading.Thread(target=print_text, daemon=True)

add_script_run_ctx(writer)

writer.start()
stream.run()
