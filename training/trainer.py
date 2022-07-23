"""Aspect and sentiment analysis training."""
import argparse
from typing import List, Optional
from pathlib import Path

from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from training.dataset import Dataset
from training.transforms import (StemSerbian, Transform, RemovePunctuation, CyrToLat, RemoveStopWords)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from training.dataset import COLUMNS


class Trainer():
    """Trainer class for aspect sentiment analysis."""
    def __init__(self, args: argparse.Namespace):
        """Constructor.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self._args = args
        self._embedding_choice = {"bag_of_words": {"embedder": CountVectorizer,
                                         "embedder_kwargs": {"ngram_range": [args.ngrams, args.ngrams],
                                                             "binary": args.binary}}}

        self._transformer_choice = {"stem": StemSerbian, "remove_punct": RemovePunctuation, 
                                    "cyr_to_lat": CyrToLat, "stop_words": RemoveStopWords}
        
                
        self._model_choice = {"logistic_regression": LogisticRegression, "naive_bayes": MultinomialNB, "svm": SVC }
        self._select_embedder()
        self._load_dataset(args.data_paths, args.exclude_value)
        self._select_transformer()
        self.transform_dataset()
        self._fit_embedder()
        self._select_model()

    def _load_dataset(self, data_paths: List[Path], exclude_value: str) -> None:
        """Load dataset.
        
        Args:
            data_paths (List[Path]): list of paths to dataset .csvs
            exclude_value str: Value representing uncategorized aspect
        """
        self._dataset = Dataset(data_paths, exclude_value)

    @property
    def dataset(self) -> Dataset:
        """Getter for dataset."""
        return self._dataset

    def _select_embedder(self) -> None:
        """Select embedder based on commmand line arguments."""
        model_class = self._embedding_choice[self._args.model_name]["embedder"]
        model_kwargs = self._embedding_choice[self._args.model_name]["embedder_kwargs"]
        self._embedder = model_class(**model_kwargs)

        if self._args.tf_idf:
            self._embedder = Pipeline([("vectorizer", self._embedder), ("tf_idf", TfidfTransformer())])

    def _fit_embedder(self) -> None:
        """Fit embedder with assigned dataset.""" # TODO: fitting should be performed only on training set
        text_np, _ = self.dataset.to_numpy()
        self._features = self._embedder.fit_transform(text_np)

    def _select_transformer(self) -> None:
        """Select transformer or combinations of transformers based on commmand line arguments."""
        transforms = [self._transformer_choice[transf] for transf in self._args.transformations]
        self._transformer = Transform(transforms)

    def transform_dataset(self) -> None:
        """Apply assigned transformations to dataset."""
        transformed_dataset = self._transformer(self.dataset)
        self._dataset = transformed_dataset
    
    def _select_model(self):
        self._model_dict = {column: self._model_choice[self._args.model]() for column in COLUMNS}
    
    def fit(self) -> None:
        _, labels_np = self.dataset.to_numpy()
        for i, column in tqdm(enumerate(COLUMNS)):
            labels_column = labels_np[:, i]
            labels_column = (labels_column != "None").astype(int)
            self._model_dict[column].fit(self._features, labels_column)
            print(f"Model trained for class: {column}")
    
    def predict(self, features:np.ndarray) -> np.ndarray:
        predictions = []
        for column in COLUMNS:
            pred = self._model_dict[column].predict(features)
            predictions.append(pred)
        
        predictions = np.array(predictions).T

        return predictions
            





