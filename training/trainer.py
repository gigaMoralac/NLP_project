import argparse
from typing import Any, List
from pathlib import Path

from training.dataset import Dataset
from pathlib import Path
from training.transforms import (StemSerbian, Transform, RemovePunctuation, CyrToLat, RemoveStopWords)
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

class Trainer():

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._embedding_choice = {"bag_of_words": {"model": CountVectorizer,
                                         "model_kwargs": {"ngram_range": [args.ngrams, args.ngrams],
                                                          "binary": args.binary}}}

        self._transformer_choice = {"stem": StemSerbian, "remove_punct": RemovePunctuation, 
                                    "cyr_to_lat": CyrToLat, "stop_words": RemoveStopWords}
        self._select_embedder()
        self._load_dataset(args.data_paths, args.exclude_value)
        self._select_transformer()
        self.transform_dataset()
        self._fit_embedder()

    def _load_dataset(self, data_paths: List[Path], exclude_value: str) -> None:
        self._dataset = Dataset(data_paths, exclude_value)

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def _select_embedder(self):
        model_class = self._embedding_choice[self._args.model_name]["model"]
        model_kwargs = self._embedding_choice[self._args.model_name]["model_kwargs"]
        self._embedder = model_class(**model_kwargs)

        if self._args.tf_idf:
            self._embedder = Pipeline([("vectorizer", self._embedder), ("tf_idf", TfidfTransformer())])

    def _fit_embedder(self):
        text_np, _ = self.dataset.to_numpy()
        self.features = self._embedder.fit_transform(text_np)

    def _select_transformer(self):
        transforms = [self._transformer_choice[transf] for transf in self._args.transformations]
        self._transformer = Transform(transforms)

    def transform_dataset(self):
        transformed_dataset = self._transformer(self.dataset)
        self._dataset = transformed_dataset

