import torch

def psnr(denoised, ground_truth):
    """
        Peak Signal to Noise Ratio: denoised and ground_truth have range [0, 1]
    """
    mse = (denoised - ground_truth).pow(2)
    mse = mse.reshape(mse.size(0), -1).mean(1)
    return -10 * mse.add_(1e-08).log10().mean()
