import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def init_weights(m: nn.Module):
    """
    Init weights with Xavier Uniform
    :param m: layer
    :return:
    """
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)

def show_batch(batch, epoch: int = 0):
    _, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.imshow(
        np.transpose(vutils.make_grid(batch[:32], padding=2, normalize=True).cpu(), (1, 2, 0))
    )
    ax.axis('off')
    plt.show()
    # plt.savefig(f'./output/images/{epoch}.png')
    # plt.close(_)
