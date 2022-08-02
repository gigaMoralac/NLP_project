"""Train aspect and sentiment model."""
import argparse
from pathlib import Path

from training.trainers import SentimentTrainer, AspectTrainer, JointTrainer
from pathlib import Path
from training.transforms import (StemSerbian, Transform, RemovePunctuation, CyrToLat, RemoveStopWords)

def add_training_arguments(parser: argparse.ArgumentParser):
    
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

if __name__ == "__main__":
    transformations = Transform([CyrToLat, RemovePunctuation, RemoveStopWords, StemSerbian])
    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    args = parser.parse_args()
    trainer = SentimentTrainer(args)
    trainer.run_pipeline()
    





    


    

