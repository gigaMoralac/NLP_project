from training.dataset import Dataset
from pathlib import Path
from training.transforms import (StemSerbian, Transform, RemovePunctuation, CyrToLat, RemoveStopWords)
data_path = Path("./dataset.csv")

if __name__ == "__main__":
    train_data = Dataset([data_path], "None")
    stemmer = StemSerbian()
    transformations = Transform([CyrToLat, RemovePunctuation, RemoveStopWords, StemSerbian])
    transformed_data = transformations(train_data)
    
    a=1

    

