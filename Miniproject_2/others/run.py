from pathlib import Path
import pickle

import torch

import Miniproject_2.others.configs as configs
from Miniproject_2.model import Model

# Define the model
model = Model()

# Load the data
train_input, train_target = torch.load(Path(configs.dataset_path) / 'train_data.pkl')
val_input, val_target = torch.load(Path(configs.dataset_path) / 'val_data.pkl')

# Train the model
model.train(train_input, train_target)

# Save the model parameters
with open('./Miniproject_2/lastmodel.pth', "wb") as f:
    pickle.dump(model.model.param(), file=f)
# Save the CPU version if in CUDA
if configs.cuda:
    params_cpu = [(p.to(device='cpu'), g.to(device='cpu')) for p, g in model.model.param()]
    with open('./Miniproject_2/lastmodel_cpu.pth', "wb") as f:
        pickle.dump(params_cpu, file=f)
