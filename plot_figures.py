# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 18:46:44 2022

@author: maxel
"""
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_epoch_losses(train_losses, test_losses, parser, plot_path):
    """ Given the average losses at each epoch of the training phase,
    Plots the evolution of the loss with respect to the number of epochs.
    The last two argument, latent_dim and num_epochs, are only osed to correctly title and save the image.
    """

    plt.figure(figsize=(6, 6))
    x = np.linspace(0, len(train_losses), len(train_losses))
    plt.plot(x, train_losses, label='training losses')
    plt.plot(x, test_losses, '--', label='test losses')
    if parser.model.lower() == 'vae':
        plt.title("Estimated average variational lower bound per epoch, $N_z$ = {}".format(parser.ls))
    else:
        plt.title("Estimated average loss per epoch, $N_z$ = {}".format(parser.ls))
    plt.xlabel("Epochs")
    plt.ylabel("Average loss per epoch")
    plt.legend()
    filename = os.path.join(plot_path, f"{parser.model}_epoch_loss_{parser.ls}d_{parser.epochs}epochs.png")
    plt.savefig(filename, dpi=100)
    plt.close()


def plot_manifold(model, device, parser, plot_path, r0=(-1, 1), r1=(-1, 1), n=12):
    w = 28
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            if parser.model.lower() == 'vae':
                mu_x, log_var_x, x_hat = model.decode(z)
            else:
                x_hat = model.decode(z)
            x_hat = x_hat.detach().cpu().numpy().reshape((28, 28))
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.xlabel('1st dim')
    plt.ylabel('2nd dim')
    plt.title('Reconstructed manifold for a 2d latent-space')
    filename = os.path.join(plot_path, f"{parser.model}_manifold_dim2_{parser.epochs}epochs.png")
    plt.savefig(filename, dpi=600)


def save_test_image(x, reconstructed_x, loss, parser, plot_path):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Image reconstructed with {} epochs and latent space of dim {}'.format(parser.epochs, parser.ls))

    ax[0] = plt.subplot(1, 2, 1)
    ax[0].set_title('original')
    ax[0].imshow(x)

    ax[1] = plt.subplot(1, 2, 2)
    ax[1].set_title('reconstructed')
    ax[1].imshow(reconstructed_x)

    plt.tight_layout()
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    filename = os.path.join(plot_path, f"test_image_{parser.model}_{parser.ls}d_{parser.epochs}epochs.png")
    plt.savefig(filename, dpi=600)
    plt.close()


def generate_image(model, device, parser, id_, dest_path, save_image=False):
    """
    Generate an image x_hat from noise as input into the trained decoder
    """
    z = torch.randn(1, parser.ls)
    z = z.to(device)
    if parser.model.lower() == 'vae':
        _, _, x_hat = model.decode(z)
    else:
        x_hat = model.decode(z)

    x_hat = x_hat.detach().cpu().numpy().reshape((28, 28))

    if save_image == True:
        plt.figure()
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        plt.imshow(x_hat)
        plt.title('Generated image from latent input')
        filename = os.path.join(dest_path, f"generate_example_{parser.ls}d_{parser.epochs}epochs_{id_}.png")
        plt.savefig(filename, dpi=600)
        plt.close()
    return x_hat


def generate_images(model, device, parser, dest_path):
    fig, ax = plt.subplots(10, 10)
    plt.rcParams["figure.figsize"] = (6, 6)

    fig.suptitle(f"Images generated with the trained decoder \n {parser.epochs} epochs, {parser.ls}d latent space")
    for i in range(1, parser.n_samples + 1):
        for j in range(1, parser.n_samples + 1):
            im = generate_image(model, device, parser, i + j, dest_path)
            ax[i - 1, j - 1].imshow(im)
            ax[i - 1, j - 1].tick_params(left=False, right=False, labelleft=False,
                                         labelbottom=False, bottom=False)
    filename = os.path.join(dest_path, f"{parser.model}_{parser.ls}d_{parser.epochs}epoch.png")
    plt.savefig(filename, dpi=100)
    plt.close()
