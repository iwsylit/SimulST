import argparse
from typing import Any, Optional, Self

from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.data.segments import SpeechSegment
from simuleval.utils import entrypoint

from simulst.audio import Audio
from simulst.models import WhisperModel
from simulst.policy import WaitkPolicy


@entrypoint
class WaitkWhisperAgent(SpeechToTextAgent):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.waitk_lagging = args.waitk_lagging
        self.source_segment_size = args.source_segment_size
        self.source_language = args.source_language
        self.target_language = args.target_language
        self.continuous_write = args.continuous_write
        self.model_size = args.model_size
        self.task = args.task

        self._model = WhisperModel(self.model_size)
        self._policy = WaitkPolicy(self.waitk_lagging, self.source_segment_size)

        if self.task == "translate":
            assert self.source_language != "en", "source language must be different from en for translation task"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        return cls(argparse.Namespace(**config))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        WaitkPolicy.add_args(parser)
        parser.add_argument("--source-language", default="en", type=str)
        parser.add_argument("--target-language", default="en", type=str)
        parser.add_argument("--model-size", default="base", type=str)
        parser.add_argument(
            "--continuous-write",
            default=1,
            type=int,
            help="""Max number of words to write at each step.
            Use negative values to write all words except N end words.
            If the prediction is longer than 15 words, only the first word is written.""",
        )
        parser.add_argument(
            "--task",
            default="transcribe",
            type=str,
            choices=["transcribe", "translate"],
        )

    def policy(self, states: Optional[AgentStates] = None) -> Action:
        if states is None:
            states = self.states  # type: ignore

        if not self._policy(states):
            return ReadAction()

        previous_translation = " ".join(states.target)
        previous_translation = previous_translation.replace(" ,", ",").replace(" .", ".")
        # TODO: fix audio saved in states to be np array instead of list
        audio = Audio.from_list(states.source)

        if audio.duration >= 28:
            self.states = self.build_states()
            self.push(
                SpeechSegment(
                    content=audio.numpy().squeeze()[-audio.sample_rate :].tolist(),
                    sample_rate=audio.sample_rate,
                    finished=False,
                )
            )

            return ReadAction()

        prediction = self._model.translate(audio, self.source_language, self.target_language, previous_translation)
        predicted_words = prediction.target.split()

        # crutch to prevent model looping
        if len(predicted_words) >= 10:
            predicted_words = predicted_words[:1]
        else:
            predicted_words = predicted_words[: self.continuous_write]

        if len(predicted_words) == 0:
            self._no_pred = True
            return ReadAction()

        return WriteAction(
            content=" ".join(predicted_words),
            finished=False,
        )
