import argparse
from abc import ABC, abstractmethod

from simuleval.agents.states import AgentStates


class Policy(ABC):
    @abstractmethod
    def __call__(self, states: AgentStates) -> bool:
        pass

    @abstractmethod
    def add_args(self, parser: argparse.ArgumentParser) -> None:
        pass


class WaitkPolicy(Policy):
    def __init__(self, waitk_lagging: int, source_segment_size: int):
        self.waitk_lagging = waitk_lagging
        self.source_segment_size = source_segment_size

    def __call__(self, states: AgentStates) -> bool:
        if states.source_sample_rate == 0:
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate

        if (length_in_seconds * 1000 / self.source_segment_size) < self.waitk_lagging:
            return False

        return True

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--waitk-lagging", default=1, type=int)
