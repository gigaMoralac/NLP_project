import argparse
from typing import Any, List
from pathlib import Path

from training.dataset import Dataset
from training.trainer import Trainer
from pathlib import Path
from training.transforms import (StemSerbian, Transform, RemovePunctuation, CyrToLat, RemoveStopWords)
import numpy as np

data_path = Path("./dataset.csv")

def add_training_arguments(parser: argparse.ArgumentParser):
    
    parser.add_argument("--model_name", type=str, choices=["bag_of_words"], default="bag_of_words")
    parser.add_argument("--ngrams", type=int, choices=[1, 2, 3])
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--tf_idf", action="store_true")
    parser.add_argument("--data_paths", nargs="+", type=Path)
    parser.add_argument("--transformations", nargs="+", type=str)
    parser.add_argument("--exclude_value", type=str, default="None")



if __name__ == "__main__":


    transformations = Transform([CyrToLat, RemovePunctuation, RemoveStopWords, StemSerbian])
    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    args = parser.parse_args()
    trainer = Trainer(args)

    a = 1








    


    

