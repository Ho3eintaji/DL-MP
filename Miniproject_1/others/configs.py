# main configs
model = 'UNet'
lr = 0.001
loss = 'l2'
batch_size = 128
gaussian_params = None
poisson_param = None
augmentations = ['exchange']
only_input_noise = False
epochs = 30
cuda = False

# demo configs
dataset_path = '../../../dataset'
train = True
validate = True
experiment_name = 'test'
save_weights = True
