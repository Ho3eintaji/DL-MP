import pickle
import random
from pathlib import Path

import torch

from .others import configs as configs
from .others import utils as utils
from .others.framework import (MSE, SGD, Conv2d, ReLU, Sequential, Sigmoid,
                               TransposeConv2d)


class Model():
    def __init__(self) -> None:
        # Define and initialize the model
        features = configs.features
        in_channels = 3
        self.model = Sequential(
            Conv2d(in_channels, features, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(features, features, kernel_size=3, stride=2, padding=1),
            ReLU(),
            TransposeConv2d(features, features, kernel_size=3, stride=2, padding=0),
            ReLU(),
            TransposeConv2d(features, in_channels, kernel_size=6, stride=2, padding=3),
            Sigmoid(),
        )

        # Move model parameters to CUDA if set
        if configs.cuda: self.model.cuda()

        # Define the optimizer and the loss function
        self.optimizer = SGD(self.model.param(), lr=configs.lr, weight_decay=configs.wd)
        self.criterion = MSE()

    def load_pretrained_model(self) -> None:
        # Loading and inserting the trained weights on the model
        with open('./Miniproject_2/bestmodel.pth', "rb") as f:
            params = (param for param in pickle.load(f))
        for module in self.model.modules:
            if isinstance(module, Conv2d) or isinstance(module, TransposeConv2d):
                module.weight, module.gradw = next(params)
                module.bias, module.gradb = next(params)

    def train(self, train_input, train_target, num_epochs=None) -> None:
        # Check the shape of the train data
        assert train_input.shape == train_target.shape

        # Set the number of epochs if not passed
        if not num_epochs:
            num_epochs = configs.epochs

        # Normalize the data
        train_input, train_target = train_input / 255., train_target / 255.

        # Load the validation data if set
        if configs.validate:
            val_input, val_target = torch.load(Path(configs.dataset_path) / 'val_data.pkl')
            if configs.cuda: val_target = val_target.cuda()

        # Train the model
        for epoch in range(num_epochs):
            # Shuffle and split the data into batches
            ix = torch.randperm(train_input.shape[0]) if configs.shuffle else torch.arange(train_input.shape[0])
            train_loader = ((train_input[ix], train_target[ix]) for ix in ix.split(configs.batch_size))

            loss_e = 0
            for img, mask in train_loader:
                # Move to CUDA if set
                if configs.cuda:
                    img = img.cuda()
                    mask = mask.cuda()

                # Flip randomly
                if configs.flip:
                    if random.random() > .5:
                        img = img.flip(2)
                        mask = mask.flip(2)
                    if random.random() > .5:
                        img = img.flip(3)
                        mask = mask.flip(3)

                # Rotate k (random) times
                if configs.rotate:
                    k = random.randint(a=0, b=3)
                    img = img.rot90(k=k, dims=(2, 3))
                    mask = mask.rot90(k=k, dims=(2, 3))

                # Train the parameters
                self.optimizer.zero_grad()
                output = self.model(img)
                loss = self.criterion(output, mask)
                loss_e += loss.item()
                # Get the gradient of the loss
                gradx, _ = self.criterion.backward()
                # Apply the gradient of the loss on the model
                self.model.backward(grad=gradx)
                # Update the parameters
                self.optimizer.step()

            if epoch % configs.patience == 0:
                if configs.validate:
                    val_psnr = utils.psnr(self.predict(val_input) / 255., val_target / 255.).item()
                    print(f'Epoch {epoch:04d} :: Training loss: {loss_e:.2e}, Validation PSNR: {val_psnr:.2f} dB')
                else:
                    print(f'Epoch {epoch:04d} :: Training loss: {loss_e:.2e}')

    def predict(self, test_input) -> torch.Tensor:
        # Normalize the data
        test_input = test_input / 255.
        # Move input to cuda if available
        if configs.cuda:
            test_input = test_input.cuda()
        # Predict the data
        out = self.model(test_input)
        return out * 255.
