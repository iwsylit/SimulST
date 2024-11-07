import argparse
from typing import Optional

from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.agents.states import AgentStates
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
        self.continuous_write = args.continuous_write
        self.model_size = args.model_size
        self.task = args.task

        self._model = WhisperModel(self.model_size)
        self._policy = WaitkPolicy(self.waitk_lagging, self.source_segment_size)

        if self.task == "translate":
            assert self.source_language != "en", "source language must be different from en for translation task"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        WaitkPolicy.add_args(parser)
        parser.add_argument("--source-language", default="en", type=str)
        parser.add_argument("--model-size", default="base", type=str)
        parser.add_argument(
            "--continuous-write",
            default=1,
            type=int,
            help="Max number of words to write at each step",
        )
        parser.add_argument(
            "--task",
            default="transcribe",
            type=str,
            choices=["transcribe", "translate"],
        )

    def policy(self, states: Optional[AgentStates] = None) -> Action:
        if states is None:
            states = self.states

        if self._policy(states):
            previous_translation = " ".join(states.target).replace(" ,", ",").replace(" .", ".")
            # TODO: fix audio saved in states to be np array instead of list
            prediction = self._model.translate(
                Audio.from_list(states.source), self.source_language, previous_translation
            )
            prediction = prediction.target.split()

            if not states.source_finished and self.continuous_write > 0:
                prediction = prediction[: self.continuous_write]

            return WriteAction(
                content=" ".join(prediction),
                finished=states.source_finished,
            )
        else:
            return ReadAction()