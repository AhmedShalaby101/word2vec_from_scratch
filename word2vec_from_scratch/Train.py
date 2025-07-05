####################Train################################
import yaml
import torch
import torch.nn as nn
from src.models.trainer import Trainer
from src.helpers import (
    get_model_class,
    get_optimizer_class,
)
from src.Utils.dataloader import get_dataloader_word2id_id2word

# reading config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# creating batches and get size of our vocab

train_dataloader, word2id, _ = get_dataloader_word2id_id2word(
    filename=config["train_dataset"],
    data_dir=config["data_dir"],
    batch_size=config["train_batch_size"],
    shuffle=config["shuffle"],
)
val_dataloader, _, _ = get_dataloader_word2id_id2word(
    filename=config["val_dataset"],
    data_dir=config["data_dir"],
    batch_size=config["val_batch_size"],
    shuffle=config["shuffle"],
)
vocab = len(word2id)

# create instances

model_class = get_model_class(config["model_name"])
model = model_class(vocab_size=vocab)

optimizer_class = get_optimizer_class(config["optimizer"])
optimizer = optimizer_class(params=model.parameters(), lr=config["learning_rate"])

# select the deivce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create instance of the trian class

trainer = Trainer(
    model=model,
    device=device,
    epochs=config["epochs"],
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    train_steps=config["train_steps"],
    val_steps=config["val_steps"],
    optimizer=optimizer,
)

trainer.training()
