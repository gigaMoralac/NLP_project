"""Train aspect and sentiment model."""
import argparse
from pathlib import Path

from training import trainers
from pathlib import Path

TRAINER_CHOICE = {"separate": ["AspectTrainer", "SentimentTrainer"],
                  "joint" : ["JointTrainer"]}


def add_training_arguments(parser: argparse.ArgumentParser):
    """Parse command-line arguments and add them to parser.

    Args:
        parser (argparse.ArgumentParser): parser to be updated.
    """ 
    parser.add_argument("--model_name", type=str, choices=["bag_of_words"], default="bag_of_words")
    parser.add_argument("--ngrams", type=int, choices=[1, 2, 3])
    parser.add_argument("--max_features", type=int, default=None) 
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--tf_idf", action="store_true")
    parser.add_argument("--data_paths", nargs="+", type=Path)
    parser.add_argument("--transformations", nargs="+", type=str)
    parser.add_argument("--exclude_value", type=str, default=None)
    parser.add_argument("--model", type=str)
    parser.add_argument("--hyperparam_range", nargs=2, metavar=("min", "max"), type=float)
    parser.add_argument("--trainer_type", type=str, choices=["separate", "joint"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    args = parser.parse_args()
    trainer_names = TRAINER_CHOICE[args.trainer_type]

    for trainer_name in trainer_names:
        current_trainer = getattr(trainers, trainer_name)(args)
        current_trainer.run_pipeline()