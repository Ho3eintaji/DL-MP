epochs = 15  # Number of epochs
lr = 5e-02  # Learning rate
wd = 0  # Weight decay
features = 32  # Number of feature channels
batch_size = 128  # Batch size
shuffle = True  # Shuffle the data on each epoch
flip = True  # Do random flips
rotate = True  # Do random rotation

cuda = False  # Use CUDA
validate = False  # Validate during the training
dataset_path = '.'  # Dataset path relative to the project root
patience = 1  # Patience for printing the loss
