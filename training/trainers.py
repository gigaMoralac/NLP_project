"""Derived trainers for aspect and sentiment analysis."""
import argparse
from .abstract_trainer import AbstractTrainer


class AspectTrainer(AbstractTrainer):
    """Train aspect models for each category."""
    def __init__(self, args: argparse.Namespace):
        """Constructor of derived class.

        Args:
            args (argparse.Namespace): command-line arguments.
        """
        self._mapping_to_labels = {"None": 0., "Positive": 1., "Negative": 1., "Neutral": 1.}
        self._f1_averaging = "macro"
        super().__init__(args)

class SentimentTrainer(AbstractTrainer):
    """Train sentiment models for each category."""
    def __init__(self, args: argparse.Namespace):
        """Constructor of derived class.

        Args:
            args (argparse.Namespace): command-line arguments.
        """
        self._mapping_to_labels = {"None": None, "Positive": 0., "Negative": 1., "Neutral": 2.}
        self._f1_averaging = "macro"
        super().__init__(args)

class JointTrainer(AbstractTrainer):
    """Train one-stage model."""
    def __init__(self, args: argparse.Namespace):
        """Constructor of derived class.

        Args:
            args (argparse.Namespace): command-line arguments.
        """
        self._mapping_to_labels = {"None": 0., "Positive": 1., "Negative": 2., "Neutral": 3.}
        self._f1_averaging = "macro"
        super().__init__(args)
