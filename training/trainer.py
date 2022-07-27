"""Aspect and sentiment analysis training."""
import argparse
from typing import Any, Dict, List, Optional, NamedTuple
from pathlib import Path
import matplotlib.pyplot as plt
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

class CrossValidation:
    def __init__(self, column: str, X: np.ndarray, y: np.ndarray, model_class: object, hyperparameter: Dict[str, List[float]]):
        self._column = column
        self.features = X
        self.labels = y
        self._model_class = model_class
        self.hyperparam_name = list(hyperparameter.keys())[0]
        self.hyperparam_values = list(hyperparameter.values())[0]
        self._metrics = None

    def run(self):
        self._metrics = []
        max_metric_value = 0
        for param in tqdm(self.hyperparam_values):
            param_kwargs = {self.hyperparam_name: param, "class_weight": "balanced"}
            model = self._model_class(**param_kwargs)
            skf = StratifiedKFold(n_splits=10, shuffle=True)
            scores = cross_val_score(model, self.features, self.labels, cv=skf, scoring="f1_macro")
            mean_score = np.mean(scores)

            if mean_score > max_metric_value:
                max_metric_value = mean_score
                best_param_value = param
            self._metrics.append((param, mean_score))

        self.best_metric = max_metric_value
        self.best_hyperparam = best_param_value
        self._store_plot()

    def _store_plot(self):
        self.figure = plt.figure();
        metrics = np.array(self._metrics)
        plt.plot(metrics[:, 0], metrics[:, 1])
        plt.xlabel(self.hyperparam_name)
        plt.ylabel("F1")
        plt.title(f"{self._model_class.__name__} : {self._column}")


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
        self._hyparameter_choice = {"logistic_regression": "C", "naive_bayes": "alpha", "svm": "C"}
        self._hyperparam_values = np.linspace(*self._args.hyperparam_range, num=10)

        self._select_embedder()
        self._load_dataset(args.data_paths, args.exclude_value)
        self._select_transformer()
        self.transform_dataset()
        self._fit_embedder()
        self._select_model()
        self._split_dataset()

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
    
    def _split_dataset(self) -> None:
        self._splits: Dict[str, Split] = {}
        _, labels_np = self.dataset.to_numpy()

        for i, column in tqdm(enumerate(COLUMNS)):
            labels_column = labels_np[:, i]
            labels_column = (labels_column != "None").astype(int)
            labels_column = labels_column
            features_train, features_test, labels_train, labels_test = train_test_split(self._features, labels_column, test_size=0.2, stratify=labels_column)
            self._splits[column] = Split(features_train, features_test, labels_train, labels_test)

    # def run_training(self) -> None:
    #     # split into train and test dataset
    #     self._splits: Dict[str, Split] = {}
    #     _, labels_np = self.dataset.to_numpy()

    #     for i, column in tqdm(enumerate(COLUMNS)):
    #         labels_column = labels_np[:, i]
    #         labels_column = (labels_column != "None").astype(int)
    #         labels_column = labels_column
    #         features_train, features_test, labels_train, labels_test = train_test_split(self._features, labels_column, test_size=0.3, stratify=labels_column)
    #         self._model_dict[column].fit(features_train, labels_train)
    #         self._splits[column] = Split(features_train, features_test, labels_train, labels_test)
    #         print(f"Model trained for class: {column}")

    def run_inference(self) -> None:

        self._scores: Dict[str, Dict[str, Any]] = {}
        for column in COLUMNS:
            test_features = self._splits[column].test_features
            test_labels = self._splits[column].test_labels
            self._scores[column] = {}
            self._scores[column]["accuracy"] = self._model_dict[column].score(test_features, test_labels)
            pred_labels = self._model_dict[column].predict(test_features)
            self._scores[column]["confusion_matrix"] = confusion_matrix(test_labels, pred_labels)
    
    def run_cross_validation(self):
        self._cvs: Dict[str, CrossValidation] = {column : None for column in COLUMNS}
        for column in COLUMNS:
            features = self._splits[column].train_features
            labels = self._splits[column].train_labels
            hyperparam_dict = {self._hyparameter_choice[self._args.model] : self._hyperparam_values}
            cv = CrossValidation(column, features, labels, self._model_choice[self._args.model], hyperparam_dict)
            cv.run()
            self._cvs[column] = cv



        


            

        








