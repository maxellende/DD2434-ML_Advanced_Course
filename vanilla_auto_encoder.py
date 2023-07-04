import random as random
import numpy as np
import torchvision
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
import os

from AE import *
from plot_figures import *
from setup_parser import create_argparser

parser = create_argparser().parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_reconstruction(image, reconstructed):
    """
    Makes a side by side comparison of an image, and it's reconstruction
    :param image:
    :param reconstructed:
    :return:
    @author: Erik Hamberg
    """
    item = image.reshape(28, 28)
    reconstructed_item = reconstructed.reshape(28, 28)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(item.detach().cpu(), cmap="binary")
    ax2.imshow(reconstructed_item.detach().cpu(), cmap="binary")
    plt.show()


def plot_latent_comparison(model, image):
    """
    This function generates a 10 by 10 grid of numbers generated by the model.
    The image entered gets encoded by the model and the encoding is slightly
    altered. The decoded versions of the altered numbers are then arranged in
    a grid.

    :param model:
    :param image:
    :return:

    @author: Erik Hamberg
    """
    image = image.reshape(-1, 28 * 28)
    code = model.encode(image)
    a = code[0][0] - 1.
    b = code[0][1] - 1.
    print(f"Changing from {code[0][0]} and {code[0][1]}, in range 0-2")
    fig, axs = plt.subplots(10, 10)
    plt.axis('off')
    for i in range(10):
        for j in range(10):
            code[0][0] = a + float(i) / 5.
            code[0][1] = b + float(j) / 5.
            constructed = model.decode(code)
            axs[i, j].imshow(constructed.reshape(28, 28).detach(), cmap="binary")
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].set_axis_off()

    fig.set_dpi(200)
    plt.show()


def plot_random_samples(model, n, ls, epoch):
    """
    Plots an n by n grid of randomly generated images.
    :param model:
    :param n:
    :return:
    """
    fig, axs = plt.subplots(n, n)
    plt.axis('off')
    for i in range(n):
        for j in range(n):
            constructed = model.generate_random()
            axs[i, j].imshow(constructed.reshape(28, 28).detach().cpu(), cmap="binary")
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].set_axis_off()

    if not os.path.exists("AE_generated_samples"):
        os.makedirs("AE_generated_samples")

    fig.set_dpi(200)
    fig.suptitle(f"{n * n} Randomly generated numbers")
    plt.savefig(os.path.join("AE_generated_samples", f"{ls}d_{epoch}epoch.png"), dpi=200)
    plt.show()

def main():
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)
    torch.cuda.manual_seed_all(parser.seed)

    transform = transforms.ToTensor()
    # rescale it to [0,1] interval later

    training_set = torchvision.datasets.MNIST('./data_mnist',
                                              download=True,
                                              train=True,
                                              transform=transform)
    test_set = torchvision.datasets.MNIST('./data_mnist',
                                          download=True,
                                          train=False,
                                          transform=transform)

    train_loader = torch.utils.data.DataLoader(training_set,
                                               batch_size=parser.batch_size,
                                               shuffle=True,
                                               num_workers=0)

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=784, latent_shape=parser.ls, hidden_dimension=parser.hd).to(device)
    # create an optimizer object
    # adagrad optimizer with learning rate 1e-3
    optimizer = torch.optim.Adagrad(model.parameters(), lr=parser.lr)

    model = get_trained_ae(model, train_loader)

    image = None

    # for i in range(5):
    #     image = test_set[i][0]
    #     image = image.reshape(-1, 28 * 28)
    #     # Output of Autoencoder
    #     reconstructed = model(image)
    #     compare_reconstruction(image, reconstructed)

    # plot_latent_comparison(model, image)
    # plot_random_samples(model, 10, LATENT_SPACE, NUM_EPOCHS)
    if parser.generate:
        generate_images(model, device, parser)


if __name__ == '__main__':
    main()