"""Transformation calsses."""
import string
from typing import Dict

import pandas as pd

from training.dataset import Dataset
from utils.StemmerByNikola import stem_str


CYR_TO_LAT_MAPPING = {'а':'a', 'б':'b', 'в':'v', 'г':'g', 'д':'d',
     'ђ':'đ', 'е':'e', 'ж':'ž', 'з':'z', 'и':'i',
     'ј':'j', 'к':'k', 'л':'l', 'љ':'q', 'м':'m',
     'н':'n', 'њ':'w', 'о':'o', 'п':'p', 'р':'r',
     'с':'s', 'т':'t', 'ћ':'ć', 'у':'u', 'ф':'f',
     'х':'h', 'ц':'c', 'џ':'y', 'ш':'š'}

class CyrToLat:
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
        out = pd.DataFrame(data=converted_text, columns=input_data.label_names)

        input_data.text = out

        return input_data

class StemSerbian:
    def __init__(self):
        """Constructor."""

    def __call__(self, input_data: Dataset) -> Dataset:
        """Stem text from input data.
        
        Args:
            input_data (Dataset): dataset to be stemmed

        """
        text_np, _ = input_data.to_numpy()
        stemmed_text = [stem_str(lines) for lines in text_np]

        out = pd.DataFrame(data=stemmed_text, columns=input_data.label_names)
        input_data.text = out

        return input_data

class RemovePunctuation:
    def __init__(self):
        """Constructor."""
    
    def __call__(self, input_data: Dataset) -> Dataset:
        """Remove punctuation from text in input data.
        
        Args:
            input_data (Dataset): Dataset from which punctuation is removed
        """
        text_np, _ = input_data.to_numpy()

        processed_text = [lines.translate(str.maketrans('', '', string.punctuation)) for lines in text_np]

        out = pd.DataFrame(data=processed_text, columns=input_data.label_names)
        input_data.text = out

        return input_data


        





