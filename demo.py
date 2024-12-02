# mypy: allow-untyped-defs

import os
import threading
import time

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from simulst.agents import WaitkWhisperAgent
from simulst.stream import StreamlitWebRtcAsrStream

TMP_FILE = "data/text.txt"

st.set_page_config(layout="wide")
st.title("Simultaneous Machine Translation Demo")

st.sidebar.title("Language Selection")

languages = ["ru", "en", "es", "fr", "de", "zh", "ja"]

source_language = st.sidebar.selectbox("Select source language", languages)
target_language = st.sidebar.selectbox("Select target language", languages[:2])


@st.cache_resource
def load_model(source_language, target_language):
    return WaitkWhisperAgent.from_dict(
        {
            "waitk_lagging": 1,
            "source_segment_size": 100,
            "source_language": source_language,
            "target_language": target_language,
            "continuous_write": -2,
            "model_size": "turbo",
            "task": "transcribe" if source_language == target_language else "translate",
        }
    )


stream = StreamlitWebRtcAsrStream(
    model=load_model(source_language, target_language),
    language=source_language,
    chunk_size=1,
    sample_rate=16000,
)


def print_text():
    while True:
        if stream.running:
            with open(TMP_FILE, "w") as f:
                f.write(stream.text)

            text_output.markdown(stream.text)
            stream_duration.markdown(f"Stream duration: {stream.stream_duration:.0f} s")

        time.sleep(0.2)


st.markdown("## Transcript")
stream_duration = st.empty()
text_output = st.empty()

print_button = st.sidebar.button("Print")
if print_button and os.path.exists(TMP_FILE):
    with open(TMP_FILE, "r") as f:
        text_output.markdown(f.read())

writer = threading.Thread(target=print_text, daemon=True)

add_script_run_ctx(writer)

writer.start()
stream.run()
