import os
import torch
from dataset import create_dataloader_demo
from network import UNet, AE
import configs as configs
import utils as utils
from tqdm import tqdm


def main():
    # Create dataset
    train_loader, val_loader = create_dataloader_demo(
        path=configs.dataset_path,
        batch_size=configs.batch_size,
        gaussian_params=configs.gaussian_params,
        poisson_param=configs.poisson_param,
        only_input_noise=configs.only_input_noise,
        augmentations=configs.augmentations,
    )

    # Model initialization
    assert configs.model in ['UNet', 'AE']
    if configs.model == 'UNet':
        model = UNet()
    else:
        model = AE()
    model = model.cuda() if configs.cuda else model

    # Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    # Creating the experiment path
    experiment_path = os.path.join('../experiments', configs.experiment_name)
    utils.create_folder(experiment_path)

    # Loss function initialization
    assert configs.loss in ['l1', 'l2']
    if configs.loss == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.L1Loss()

    # Training
    if configs.train:
        for epoch in range(configs.epochs):
            model.train()

            for img, target in tqdm(train_loader, position=0, leave=True):
                img = img.cuda().float() if configs.cuda else img.float()
                target = target.cuda() if configs.cuda else target

                optimizer.zero_grad()

                # Backward propagation
                output = model(img)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Validation
            if configs.validate:
                model.eval()
                with torch.no_grad():
                    for img, target in val_loader:
                        img = img.cuda().float() if configs.cuda else img.float()
                        target = target.cuda() if configs.cuda else target

                        output = model(img)
                        loss = criterion(output, target)
                        utils.record(loss, output, target)

                # Logging
                utils.log_track(epoch)

                # Saving the loss and the scores of validation for this epoch
                utils.save_track(experiment_path)

            # Saving the weights
            if configs.save_weights:
                utils.save_model(model, optimizer, experiment_path)


if __name__ == '__main__':

    # Getting and validating the configurations
    if configs.cuda:
        if not torch.cuda.is_available():
            print("GPU not available. Using CPU.")
            configs.cuda = False

    main()
