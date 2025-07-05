############## Trainer #########################
import torch
import numpy as np
import torch.nn as nn


class Trainer:
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        val_dataloader,
        train_steps,
        val_steps,
        optimizer,
        device,
    ):
        self.epochs = epochs  # from config file
        self.train_dataloader = train_dataloader  # from class dataloader
        self.val_dataloader = val_dataloader  # from class dataloader
        self.train_steps = train_steps  # from config file
        self.val_steps = val_steps  # from config file
        self.model = model  # create instance in train.py
        self.optimizer = optimizer  # create instance in train.py
        self.device = device  # checked in the train.py

        self.criterion = nn.CrossEntropyLoss()  # fully defined
        self.model.to(self.device)  # adjust the model to the device
        self.loss = {"train": [], "val": []}

    def training(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            self.val_epoch()
            print(
                f"Epoch:{epoch + 1}/{self.epochs},train_loss = {self.loss['train'][-1]:.4f} , val_loss = {self.loss['val'][-1]:.4f}"
            )
        print("training has been completed :)")

    def train_epoch(self):
        running_loss = []
        self.model.train()

        for step, batch in enumerate(self.train_dataloader, start=1):
            # selecting batch & adjust to device
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # forward propagation
            self.optimizer.zero_grad()
            outputs_pred = self.model(inputs)
            loss = self.criterion(outputs_pred, targets)

            # backward propagation
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if step == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def val_epoch(self):
        running_loss = []
        self.model.eval()

        with torch.inference_mode():  # turns off gradient tracking
            for step, batch in enumerate(self.val_dataloader, start=1):
                # selecting batch & adjust to device
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # forward propagation
                outputs_pred = self.model(inputs)
                loss = self.criterion(outputs_pred, targets)
                running_loss.append(loss.item())

                if step == self.val_steps:
                    break

            epoch_loss = np.mean(running_loss)
            self.loss["val"].append(epoch_loss)
