"""Class for dataset manipulation."""
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

COLUMNS = ["Opsti utisak film", "Radnja", "Rezija", "Gluma","Muzika",
           "Opsti utisak komentar", "Specijalni efekti"]

SUPPORTED_FILE_TYPES = [".csv", ".tsv"]

class Dataset():
    def __init__(self, files: List[Path], exclude_value: Optional[str], 
                 label_names: List[str] = COLUMNS):
        """Constructor for dataset class.

        Args:
            files (List[Path]): paths to the .csv files containing parts of annotated dataset.
        """
        self.label_names = label_names
        text_container: List[pd.DataFrame] = []
        label_container: List[pd.DataFrame] = []

        for file in files:
            assert file.exists(), f"Provided file: {file} doesn't exist"
            texts, labels = self._read_data_from_file(file, exclude_value)
            text_container.append(texts)
            label_container.append(labels)

        self.text = pd.concat(text_container)
        self.labels = pd.concat(label_container)

    def _read_data_from_file(self, file_path: Path,
                              exclude_value: Optional[str] = "None") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read texts and labels corresponding to them.
        Args:
            file_path (Path): path to .csv file containing text data and labels
            exclude_value (str): exclude rows where all columns have this value
        """
        assert file_path.suffix in SUPPORTED_FILE_TYPES, \
            f"Provided file extension: {file_path.suffix} is unsupported"

        data = pd.read_csv(file_path)
        labels = data[self.label_names]
        texts = data[["text"]]
        if exclude_value is not None:
            valid_indices = self.index_rows_based_on_value(labels, exclude_value)
            labels = labels[valid_indices]
            texts = texts[valid_indices]

        return texts, labels

    @staticmethod
    def index_rows_based_on_value(data: pd.DataFrame,
                                   exclude_value: str) -> List[bool]:
        """Return bool array which will have valu

        Args:
            data (pd.DataFrame): input dataframe.
            exclude_value (str): input value.
        """

        # compare each value with exlcude_value
        comparasion: pd.DataFrame = data == exclude_value

        # cast it to ndarray for easier manipulation
        comparasion = comparasion.to_numpy()

        # return logic array which has True values for rows with specific value
        indices = np.all(comparasion, axis=1)

        return np.invert(indices).tolist()
    
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return text and labels as numpy array."""
        return self.text.to_numpy().flatten(), self.labels.to_numpy()
