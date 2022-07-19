from training import dataset
from pathlib import Path

data_path = Path("./dataset.csv")
if __name__ == "__main__":
    train_data = dataset.Dataset([data_path], "None")
    

