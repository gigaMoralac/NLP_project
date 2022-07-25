"""Aspect and sentiment analysis training."""
import argparse
from typing import Any, Dict, List, Optional, NamedTuple
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class Split(NamedTuple):
    train_features: np.ndarray
    test_features: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray

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
        
                
        self._model_choice = {"logistic_regression": LogisticRegression, "naive_bayes": MultinomialNB, "svm": SVC}
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
        embedder_class = self._embedding_choice[self._args.model_name]["embedder"]
        embedder_kwargs = self._embedding_choice[self._args.model_name]["embedder_kwargs"]
        self._embedder = embedder_class(**embedder_kwargs)

        if self._args.tf_idf:
            self._embedder = Pipeline([("vectorizer", self._embedder), ("tf_idf", TfidfTransformer())])

    def _fit_embedder(self) -> None:
        """Fit embedder with assigned dataset.""" # TODO: fitting should be performed only on training set
        text_np, _ = self.dataset.to_numpy()
        self._features = self._embedder.fit_transform(text_np).toarray()

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
    
    # def fit(self) -> None:
    #     _, labels_np = self.dataset.to_numpy()
    #     for i, column in tqdm(enumerate(COLUMNS)):
    #         labels_column = labels_np[:, i]
    #         labels_column = (labels_column != "None").astype(int)
    #         self._model_dict[column].fit(self._features, labels_column)
    #         print(f"Model trained for class: {column}")
    
    def predict(self, features:np.ndarray) -> np.ndarray:
        predictions = []
        for column in COLUMNS:
            pred = self._model_dict[column].predict(features)
            predictions.append(pred)
        
        predictions = np.array(predictions).T

        return predictions
    
    def run_training(self) -> None:
        # split into train and test dataset
        self._splits: Dict[str, Split] = {}
        _, labels_np = self.dataset.to_numpy()

        for i, column in tqdm(enumerate(COLUMNS)):
            labels_column = labels_np[:, i]
            labels_column = (labels_column != "None").astype(int)
            labels_column = labels_column
            features_train, features_test, labels_train, labels_test = train_test_split(self._features, labels_column, test_size=0.3, stratify=labels_column)
            self._model_dict[column].fit(features_train, labels_train)
            self._splits[column] = Split(features_train, features_test, labels_train, labels_test)
            print(f"Model trained for class: {column}")

    def run_inference(self) -> None:

        self._scores: Dict[str, Dict[str, Any]] = {}
        for column in COLUMNS:
            test_features = self._splits[column].test_features
            test_labels = self._splits[column].test_labels
            self._scores[column] = {}
            self._scores[column]["accuracy"] = self._model_dict[column].score(test_features, test_labels)
            pred_labels = self._model_dict[column].predict(test_features)
            self._scores[column]["confusion_matrix"] = confusion_matrix(test_labels, pred_labels)


            

        








