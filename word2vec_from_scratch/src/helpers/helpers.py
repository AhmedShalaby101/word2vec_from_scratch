#############################Helpers###############################################
import yaml
import torch
import torch.optim as optim
from src.Utils.dataloader import CBOW


def get_model_class(model_name: str):
    if model_name == "CBOW":
        return CBOW
    else:
        raise ValueError("Only Available model is CBOW")
        return


def get_optimizer_class(optimizer: str):
    if optimizer == "Adam":
        return optim.Adam
    else:
        raise ValueError("Only Available optimizer is Adam")
        return
