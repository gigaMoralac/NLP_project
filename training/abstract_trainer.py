"""Aspect and sentiment analysis training."""
import argparse
from abc import ABCMeta
import json
import logging
from more_itertools import one
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm
from utils.pathutils import create_dir_hierarchy

from training.dataset import COLUMNS, Dataset
from training.transforms import (CyrToLat, RemovePunctuation, RemoveStopWords,
                                 StemSerbian, Transform)


class CrossValidation:
    """Class for performing cross validation and storing data related to it.
       Each instance should be aspect-specific.
    """
    def __init__(self, column: str, model_class: object,
                 hyp_kwargs_list: List[Dict[str, float]]): 
        """Constructor.
        
        Args:
            column (str): aspect for which the cross-validation should be performed.
            model_class (object): class of the model used for training.
            hyp_kwargs_list (List[Dict[str, float]]): List of hyperparameter kwargs.
            
        """
        self._column = column
        self._model_class = model_class
        self._skf = StratifiedKFold(n_splits=10, shuffle=True)
        self._hyperparam_kwargs_list = hyp_kwargs_list
        self.hyperparam_name = one(hyp_kwargs_list[0].keys())
        self._metrics = None

    def run(self, features: np.ndarray, labels: np.ndarray):
        """Run cross validation.

        Args:
            features (np.ndarray): input features.
            labels (np.ndarray): input labels (ground truth).
        """
        self._metrics = []
        max_metric_value = 0
        for hyperparam_kwargs in tqdm(self._hyperparam_kwargs_list):
            param = one(hyperparam_kwargs.values())
            hyperparam_kwargs["class_weight"] = "balanced"
            model = self._model_class(**hyperparam_kwargs)
            scores = cross_val_score(model, features, labels, cv=self._skf, scoring="f1_macro")
            mean_score = np.mean(scores)

            if mean_score > max_metric_value:
                max_metric_value = mean_score
                best_param_value = param
            self._metrics.append((param, mean_score))

        self.best_metric = max_metric_value
        self.best_hyperparam = best_param_value
        self._store_plot()

    def _store_plot(self):
        """Store plot for each cross-validation curve."""
        self.figure = plt.figure()
        metrics = np.array(self._metrics)
        plt.plot(metrics[:, 0], metrics[:, 1])
        plt.xlabel(self.hyperparam_name)
        plt.ylabel("F1")
        plt.title(f"{self._model_class.__name__} : {self._column}")
        plt.close()


class Split(NamedTuple):
    """Container for data from dataset split."""
    train_features: np.ndarray
    test_features: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray

class AbstractTrainer(metaclass=ABCMeta):
    """Base trainer class for aspect and sentiment analysis."""
    def __init__(self, args: argparse.Namespace):
        """Base Constructor.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self._args = args
        self._embedding_choice = {"bag_of_words": {"embedder": CountVectorizer,
                                  "embedder_kwargs": {"ngram_range": [args.ngrams, args.ngrams],
                                                      "binary": args.binary,
                                                      "max_features": args.max_features}}}

        self._transformer_choice = {"stem": StemSerbian, "remove_punct": RemovePunctuation,
                                    "cyr_to_lat": CyrToLat, "stop_words": RemoveStopWords}
             
        self._model_choice = {"logistic_regression": LogisticRegression,
                              "naive_bayes": MultinomialNB,
                              "svm": SVC}
        self._hyparameter_choice = {"logistic_regression": "C", "naive_bayes": "alpha", "svm": "C"}
        self._hyperparam_values = np.linspace(*self._args.hyperparam_range, num=10)

        self._select_embedder()
        self._load_dataset(args.data_paths, args.exclude_value)
        self._select_transformer()
        self.transform_dataset()
        self._fit_embedder()
        self._split_dataset()

    def _load_dataset(self, data_paths: List[Path], exclude_value: str) -> None:
        """Load dataset.
        
        Args:
            data_paths (List[Path]): list of paths to dataset .csvs
            exclude_value str: Value representing uncategorized aspect
        """
        self._dataset = Dataset(data_paths, exclude_value)
    
    def _save_run_args(self, write_dir: Path):
        """Write comand-line arguments to .json file
        
        Args:
            write_dir (Path): directory where .json file will be written
        """
        args_as_dict = vars(self._args)
        # we have to convert args of type pathlib.Path into str in order for it to be json serializable
        args_as_dict["data_paths"] = [str(path) for path in args_as_dict["data_paths"]]

        json_file_path = write_dir / "args.json"
        assert write_dir.exists(), f"Provided directory {write_dir} doesn't exist"
        with open(json_file_path, "w") as json_file:
            json.dump(args_as_dict, json_file, indent=2)
            json_file.close()
        
        logging.info(f"\nRunning arguments saved at {str(json_file_path)}\n")

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
    
    def _init_models(self, hyperparam_kwargs: Dict[str, Dict[str, Any]]):
        """Initialize models for each aspect with provided hyperparams.
        
        Args:
            hyperparam_kwargs (Dict[str, Dict[str, Any]]): kwargs containing hyperparams
        """
        self._model_dict = {column: self._model_choice[self._args.model](**hyperparam_kwargs[column])
                            for column in COLUMNS}
    
    def _split_dataset(self) -> None:
        """Split dataset to train and test set for each aspect."""
        self._splits: Dict[str, Split] = {}
        _, labels_np = self.dataset.to_numpy()

        for i, column in tqdm(enumerate(COLUMNS)):
            labels_column = labels_np[:, i]
            labels_column = np.array([self._mapping_to_labels[label] for label in labels_column])
            valid_indices = labels_column != None
            data_splits = train_test_split(self._features[valid_indices],
                                           labels_column[valid_indices],
                                           test_size=0.2,
                                           stratify=labels_column[valid_indices])

            features_train, features_test, labels_train, labels_test = data_splits 
            labels_train = labels_train.astype(np.float32)
            labels_test = labels_test.astype(np.float32)
            self._splits[column] = Split(features_train, features_test, labels_train, labels_test)

    def run_training(self) -> None:
        """Run training for each aspect with hyperparams determined in cross validation."""
        # split into train and test dataset
        hyperparams_per_class = {column: {self._hyparameter_choice[self._args.model]: self._cvs[column].best_hyperparam}
                                 for column in COLUMNS}
        self._init_models(hyperparams_per_class)
        for column in self._splits.keys():

            self._model_dict[column].fit(self._splits[column].train_features, self._splits[column].train_labels)
            pred_labels = self._model_dict[column].predict(self._splits[column].train_features)
            logging.info(f"Model trained for class: {column}")
            score = f1_score(self._splits[column].train_labels, pred_labels, average=self._f1_averaging)
            confusion_mat = confusion_matrix(self._splits[column].train_labels, pred_labels)
            logging.info(f"Train score: {score}")
            logging.info(f"Confusion_matrix: \n{confusion_mat}")
            
    def run_inference(self) -> None:
        """Perform inference on trained model."""
        self._scores: Dict[str, Dict[str, Any]] = {}
        for column in COLUMNS:
            test_features = self._splits[column].test_features
            test_labels = self._splits[column].test_labels
            self._scores[column] = {}
            logging.info(f"Model inference for class: {column}")
            pred_labels = self._model_dict[column].predict(test_features)
            score = f1_score(test_labels, pred_labels, average=self._f1_averaging)
            confusion_mat = confusion_matrix(test_labels, pred_labels)

            logging.info(f"Test score : {score}")
            logging.info(f"Confusion matrix on test set: \n{confusion_mat}")

            self._scores[column]["f1_score"] = score
            self._scores[column]["confusion_matrix"] = confusion_mat
    
    def run_cross_validation(self):
        """Run cross-validation with previously computed data splits."""
        self._dirs.log_cv.mkdir(parents=True, exist_ok=True)
        self._cvs: Dict[str, CrossValidation] = {column : None for column in COLUMNS}
        for column in COLUMNS:
            features = self._splits[column].train_features
            labels = self._splits[column].train_labels

            hyperparam_name = self._hyparameter_choice[self._args.model]
            hyperparam_kwargs_list = [{hyperparam_name : value} for value in  self._hyperparam_values]
            cv = CrossValidation(column, self._model_choice[self._args.model], hyperparam_kwargs_list)

            logging.info(f"cross validating for class: {column}")
            cv.run(features, labels)
            logging.info(f"cross_validating for class {column} completed")
            logging.info(f"Optimal value of parameter {cv.hyperparam_name}: {cv.best_hyperparam}")
            logging.info(f"Optimal metric value: {cv.best_metric}")

            # saving plot image
            figure_name = f"{column}.png"
            cv.figure.savefig(self._dirs.log_cv / figure_name) 
            logging.info(f"cross validation graph saved to {str(self._dirs.log_cv / figure_name)}")
            self._cvs[column] = cv
    
    def run_pipeline(self):
        """Runs the whole training, validation and evaluation pipeline."""
        self._dirs = create_dir_hierarchy()
        self._dirs.log_dir.mkdir(parents=True, exist_ok=True)
        self._save_run_args(self._dirs.log_dir)
        logging.basicConfig(filename=self._dirs.log_dir / "log.log", level=logging.INFO)
        logging.info(f"trainer_type : {self.__class__.__name__}")
        logging.info(f"model: {self._args.model}")

        logging.info("Cross-Validation started")

        self.run_cross_validation()

        logging.info("Cross-Validation finished")

        logging.info("Training started")
        self.run_training()
        logging.info("Training finished")

        logging.info("Inference started")
        self.run_inference()
        logging.info("Inference finished")
