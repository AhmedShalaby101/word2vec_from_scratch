##import libraries
from datasets import load_dataset
import pandas as pd

##load the dataset
dataset = load_dataset("taaredikahan23/medical-llama2-5k")

df = pd.DataFrame(dataset["train"]["text"])


##export dataset
df.to_pickle("../../data/raw/raw_dataset.pkl")
