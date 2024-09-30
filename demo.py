# mypy: allow-untyped-defs

import threading
import time

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from simulst.models import WhisperModel
from simulst.stream import StreamlitWebRtcAsrStream

st.set_page_config(layout="wide")
st.title("Simultaneous Machine Translation")

st.sidebar.title("Language Selection")

languages = ["ru", "en", "es", "fr", "de", "zh", "ja"]

source_language = st.sidebar.selectbox("Select source language", languages)
target_language = st.sidebar.selectbox("Select target language", languages)

whisper_model = WhisperModel("openai/whisper-tiny")
stream = StreamlitWebRtcAsrStream(
    model=whisper_model, language=source_language, chunk_size=2, overlap_size=1, buffer_size=5, sample_rate=16000
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
