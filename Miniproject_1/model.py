# For mini-project 1
from pathlib import Path
import torch
from Miniproject_1.others.dataset import create_dataloader
from Miniproject_1.others.network import UNet
import Miniproject_1.others.configs as configs


class Model():
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(device=self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configs.lr)

        assert configs.loss in ['l1', 'l2']
        if configs.loss == 'l2':
            self.criterion = torch.nn.MSELoss().to(device=self.device)
        else:
            self.criterion = torch.nn.L1Loss().to(device=self.device)

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model pass

        try:
            model_path = Path(__file__).parent / "bestmodel.pth"
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            return

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self, train_input, train_target, num_epochs=None) -> None:
        # :train_input: tensor of size (N, C, H, W) containing a noisy version of the images. :train_target: tensor
        # of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input
        # by their noise.
        if num_epochs is None:
            num_epochs = configs.epochs

        train_input = train_input.float()
        train_target = train_target.float()
        # Normalize train data in case it's not normalized
        if torch.max(train_input) > 1.0:
            train_input = train_input / 255.0
        if torch.max(train_target) > 1.0:
            train_target = train_target / 255.0

        # Create dataloader
        train_loader = create_dataloader(
            x=train_input,
            y=train_target,
            batch_size=configs.batch_size,
            gaussian_params=configs.gaussian_params,
            poisson_param=configs.poisson_param,
            augmentations=configs.augmentations
        )

        self.model.train()

        # Just to make sure gradient is enabled for this miniproject during training
        torch.set_grad_enabled(True)
        for epoch in range(num_epochs):
            for img, target in train_loader:
                img = img.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()

                # Backward propagation
                output = self.model(img)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def predict(self, test_input) -> torch.Tensor:
        # :test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        # : returns a tensor of the size (N1, C, H, W)

        test_input = test_input.float()
        # Normalize test data in case it's not normalized
        if torch.max(test_input) > 1.0:
            test_input = test_input / 255.0
        self.model.eval()
        with torch.no_grad():
            img = test_input.to(self.device)
            out = self.model(img)
        out = torch.clip(out * 255, min=0, max=255)
        return out
