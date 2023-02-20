import os
import pandas as pd
import torch

records = {'loss': [], 'psnr': []}
results = {'loss': [], 'psnr': []}
max_res = None


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def psnr(denoised, ground_truth):
    """
        Peak Signal to Noise Ratio: denoised and ground_truth have range [0, 1]
    """
    mse = (denoised - ground_truth) ** 2
    mse = mse.view(mse.size(0), -1).mean(1)
    return torch.mean(-10 * torch.log10(mse + 10 ** -8))


def record(loss, denoised, ground_truth):
    """
        Record the results in each epoch.
    """
    records['psnr'].append(psnr(denoised, ground_truth).item())
    records['loss'].append(loss.item())


def save_track(path):
    """
        Saves the result of the epoch in the corresponding path.
    """
    results['loss'].append(sum(records['loss']) / len(records['loss']))
    results['psnr'].append(sum(records['psnr']) / len(records['psnr']))
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(path, "results.csv"))
    records['loss'] = []
    records['psnr'] = []


def log_track(epoch):
    print('Epoch : {} | Loss = {:.4f}, PSNR = {:.4f}'.format(
        epoch,
        sum(records['loss']) / len(records['loss']),
        sum(records['psnr']) / len(records['psnr'])
    )
    )


def save_model(model, optimizer, path):
    """
        Saves the model and the optimizer states.
    """
    global max_res
    if max_res is None or results['psnr'][-1] > max_res:
        max_res = results['psnr'][-1]
        save_path = os.path.join(path, 'model.pt')
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, save_path)
