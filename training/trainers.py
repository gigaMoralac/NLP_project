import argparse

from .abstract_trainer import AbstractTrainer


class AspectTrainer(AbstractTrainer):
    def __init__(self, args: argparse.Namespace):
        self._mapping_to_labels = {"None": 0., "Positive": 1., "Negative": 1., "Neutral": 1.}
        self._f1_averaging = "binary"
        super().__init__(args)

class SentimentTrainer(AbstractTrainer):
    def __init__(self, args: argparse.Namespace):
        self._mapping_to_labels = {"None": None, "Positive": 0., "Negative": 1., "Neutral": 2.}
        self._f1_averaging = "weighted"
        super().__init__(args)

class JointTrainer(AbstractTrainer):
    def __init__(self, args: argparse.Namespace):
        self._mapping_to_labels = {"None": 0., "Positive": 1., "Negative": 2., "Neutral": 3.}
        self._f1_averaging = "weighted"
        super().__init__(args)
