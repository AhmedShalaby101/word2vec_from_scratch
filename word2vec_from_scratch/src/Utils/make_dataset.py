##import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

##load dataset
df = pd.read_pickle("../../data/raw/raw_dataset.pkl")

## create train and val data

train_dataset, val_dataset = train_test_split(
    df["text"], test_size=0.2, random_state=42
)
train_dataset = pd.DataFrame(train_dataset)
val_dataset = pd.DataFrame(val_dataset)
##export the datasets

train_dataset.to_pickle("../../data/processed/train_dataset")
val_dataset.to_pickle("../../data/processed/val_dataset")
