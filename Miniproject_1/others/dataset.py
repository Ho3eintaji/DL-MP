from abc import ABC, abstractmethod
import torch
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision.transforms.functional as TF


class NoiseGenerator(ABC):

    @abstractmethod
    def add_train_noise(self, img):
        pass

    @abstractmethod
    def add_val_noise(self, img):
        pass


class GaussianNoiseGenerator(NoiseGenerator):

    def __init__(self, train_max_std, train_min_std, val_std):
        self.train_max_std, self.train_min_std = train_max_std, train_min_std
        self.val_std = val_std

    def add_train_noise(self, img):
        # Randomly select the variance from the given range
        std = torch.FloatTensor(1).uniform_(self.train_max_std / 255, self.train_min_std / 255)
        # Generate and add noise on the training image
        return img + torch.randn_like(img) * std

    def add_val_noise(self, img):
        # Generate and add noise on the validation image
        return img + torch.randn_like(img) * self.val_std / 255


class PoissonNoiseGenerator(NoiseGenerator):

    def __init__(self, param_max):
        self.param_max = param_max

    def add_train_noise(self, img):
        # Randomly select the parameter from the given range
        param_range = torch.FloatTensor(1).uniform_(0.001, self.param_max)
        # Generate the noisy image
        return torch.poisson(param_range * (img + 0.5)) / param_range - 0.5

    def add_val_noise(self, img):
        # Generate the noisy image
        param_range = 30
        return torch.poisson(param_range * (img + 0.5)) / param_range - 0.5


class NoisySet(Dataset):

    def __init__(self, x, y, set_type, noise_generator=None, only_input_noise=False, augmentations=[]):
        super(Dataset, self).__init__()

        assert set_type in ['train', 'val']

        self.set_type = set_type
        self.noise = noise_generator
        self.rotate = 'rotate' in augmentations
        self.flip = 'flip' in augmentations
        self.exchange = 'exchange' in augmentations
        self.only_input_noise = only_input_noise

        self.pairs = (x, y)

    def transform(self, img, target):
        if self.flip:
            if random.random() > 0.5:
                img = TF.hflip(img)
                target = TF.hflip(target)
            if random.random() > 0.5:
                img = TF.vflip(img)
                target = TF.vflip(target)

        if self.rotate:
            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
            target = TF.rotate(target, angle)

        if self.exchange:
            # Randomly select pixels that will be exchanged
            w1, w2 = torch.rand_like(img), torch.rand_like(img)
            # Exchange the pixels between the image and the target
            img, target = w1 * img + (1-w1) * target, w2 * img + (1-w2) * target

        return img, target

    def __getitem__(self, index):
        img1, img2 = self.pairs[0][index], self.pairs[1][index]

        # Apply noise if needed
        if self.noise:
            if self.set_type == 'train':
                img1 = self.noise.add_train_noise(img1)
                # Apply noise also on the target in case only_input_noise is False
                if not self.only_input_noise:
                    img2 = self.noise.add_train_noise(img2)
            else:
                img1 = self.noise.add_val_noise(img1)
        img1, img2 = self.transform(img1, img2)

        return img1, img2

    def __len__(self):
        return self.pairs[0].shape[0]


def create_dataloader_demo(path, batch_size=4, gaussian_params=(0, 50, 25), poisson_param=50, only_input_noise=False, augmentations=[]):
    # Create noise generator object if additional noise is needed
    noise_gen = None
    if gaussian_params:
        train_max_std, train_min_std, val_std = gaussian_params
        noise_gen = GaussianNoiseGenerator(train_max_std, train_min_std, val_std)
    elif poisson_param:
        noise_gen = PoissonNoiseGenerator(poisson_param)

    # Load training data and create training dataset
    train_images_path = Path(path) / 'train_data.pkl'
    x, y = torch.load(train_images_path)
    train_set = NoisySet(x[:128] / 255.0, y[:128] / 255.0, set_type='train', noise_generator=noise_gen, only_input_noise=only_input_noise, augmentations=augmentations)

    # Load validation data and create validation dataset
    valid_images_path = Path(path) / 'val_data.pkl'
    x, y = torch.load(valid_images_path)
    val_set = NoisySet(x / 255.0, y / 255.0, set_type='val', noise_generator=noise_gen)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

    return train_loader, val_loader


def create_dataloader(x, y, batch_size=4, gaussian_params=(0, 50, 25), poisson_param=50, only_input_noise=False, augmentations=[]):
    # Create noise generator object if additional noise is needed
    noise_gen = None
    if gaussian_params:
        train_max_std, train_min_std, val_std = gaussian_params
        noise_gen = GaussianNoiseGenerator(train_max_std, train_min_std, val_std)
    elif poisson_param:
        noise_gen = PoissonNoiseGenerator(poisson_param)

    # Create training dataset
    train_set = NoisySet(x, y, set_type='train', noise_generator=noise_gen, only_input_noise=only_input_noise, augmentations=augmentations)

    return DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    path = '../../../dataset'
    x, y = torch.load(path + '/train_data.pkl')
    train_loader = create_dataloader(x / 255, y / 255, only_input_noise=True, augmentations=['rotate', 'flip'])
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)

    train_loader, val_loader = create_dataloader_demo(path, only_input_noise=True, augmentations=['rotate', 'flip'])
    x, y = next(iter(train_loader))
    x, y = next(iter(val_loader))
    print(x.shape, y.shape)
