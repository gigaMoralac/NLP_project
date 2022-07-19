"""Transformation calsses."""
from pathlib import Path
import string
from typing import Callable, Dict, List

import pandas as pd
import numpy as np

from training.dataset import Dataset
from utils.StemmerByNikola import stem_str


CYR_TO_LAT_MAPPING = {'а':'a', 'б':'b', 'в':'v', 'г':'g', 'д':'d',
     'ђ':'đ', 'е':'e', 'ж':'ž', 'з':'z', 'и':'i',
     'ј':'j', 'к':'k', 'л':'l', 'љ':'q', 'м':'m',
     'н':'n', 'њ':'w', 'о':'o', 'п':'p', 'р':'r',
     'с':'s', 'т':'t', 'ћ':'ć', 'у':'u', 'ф':'f',
     'х':'h', 'ц':'c', 'џ':'y', 'ш':'š'}

STOP_WORD_DICT_PATH = Path("./assets/stop_words_serbian.txt")


class CyrToLat():
    def __init__(self, mapping: Dict[str, str] = CYR_TO_LAT_MAPPING):
        """Constructor.
 
        Args:
            mapping (Dict[str, str]): mapping from cyrilic to latin 
        """
        self.mapping = mapping
        self.translation_table = str.maketrans(self.mapping)

    def __call__(self, input_data: Dataset) -> Dataset:
        """Convert text from cyrlic to latin.
        
        Args:
            input_data (Dataset): dataset to be transformed.
        """

        translation_table = str.maketrans(self.mapping)
        text_np, _ = input_data.to_numpy()

        converted_text = [lines.translate(translation_table) for lines in text_np]
        out = pd.DataFrame(data=converted_text, columns=["text"])

        input_data.text = out

        return input_data


class StemSerbian():
    def __init__(self):
        """Constructor."""

    def __call__(self, input_data: Dataset) -> Dataset:
        """Stem text from input data.

        Args:
            input_data (Dataset): dataset to be stemmed

        """
        text_np, _ = input_data.to_numpy()
        stemmed_text = [stem_str(lines) for lines in text_np]

        out = pd.DataFrame(data=stemmed_text, columns=["text"])
        input_data.text = out

        return input_data


class RemovePunctuation():
    def __init__(self):
        """Constructor."""

    def __call__(self, input_data: Dataset) -> Dataset:
        """Remove punctuation from text in input data.
        
        Args:
            input_data (Dataset): Dataset from which punctuation is removed
        """
        text_np, _ = input_data.to_numpy()

        processed_text = [lines.translate(str.maketrans('', '', string.punctuation)) for lines in text_np]

        out = pd.DataFrame(data=processed_text, columns=["text"])
        input_data.text = out

        return input_data


class RemoveStopWords():
    """Class for removing stop words from dataset."""
    def __init__(self, stopwords_path: Path=STOP_WORD_DICT_PATH):
        """Constructor.
        Args:
            stopwords_path (Path): Path to stop-words dictionary.
        """
        self.stopword_list = np.loadtxt(stopwords_path, dtype=str)

    def __call__(self, input_data: Dataset) -> Dataset:
        """Remove stop-words from input data.
        
        Args:
            input_data (Dataset): Dataset from which the stopwords will be removed
        """
        text_np, _ = input_data.to_numpy()
        processed_text = []
        for lines in text_np:
            segmented_line = [word for word in lines.lower().split() if word not in self.stopword_list]
            segmented_line = " ".join(segmented_line)
            processed_text.append(segmented_line)

        out = pd.DataFrame(data=processed_text, columns=["text"])
        input_data.text = out

        return input_data


class Transform():
    """Class for chaining Multiple transformations on Dataset"""
    def __init__(self, transformations: List[Callable[[Dataset], Dataset]]):
        """Constructor.
        
        Args:
            transformations (List[Callable[[Dataset], Dataset]]): List of transformations to be chained.
        """
        self.transforms = transformations

    def __call__(self, data: Dataset) -> Dataset:
        """Chain transformations and apply them to input data.
        
        Args:
            data (Dataset): input data.
        """
        for transform in self.transforms:
            t = transform()
            data = t(data)
        
        return data


